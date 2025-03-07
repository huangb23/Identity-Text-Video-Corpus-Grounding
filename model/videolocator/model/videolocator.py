import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from videolocator.span_utils import span_xx_to_cxw, span_cxw_to_xx, generalized_temporal_iou, temporal_iou
from videolocator.constants import VIDEO_TOKEN_INDEX, PERSON_TOKEN_INDEX
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import get_peft_model

class VideoLocatorConfig(LlamaConfig):
    model_type = "VideoLocator"

class VideoLocatorLlamaModel(LlamaModel):
    config_class = VideoLocatorConfig

    def grounding_forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_type: str = 'none'
    ) -> Union[Tuple, BaseModelOutputWithPast]:


        batch_size, seq_length, _ = inputs_embeds.shape

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, 0
        )



        hidden_states = inputs_embeds


        mid_layer = 16
        if input_type == 'text' or input_type == 'video':
            layers = self.layers[:mid_layer]
        elif input_type == 'grounding':
            layers = self.layers[mid_layer:]
        else:
            print("Error in input_type")

        for decoder_layer in layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, False, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
       
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states
    

class VideoLocator(LlamaForCausalLM):
    config_class = VideoLocatorConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        hidden_size = config.hidden_size
        clip_size = 768
        face_size = 512
        cont_size = 768

        self.vid_input_projector = nn.Sequential(
            nn.Linear(clip_size, hidden_size), 
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.vid_face_input_projector = nn.Linear(face_size, hidden_size)
        self.vid_proj = nn.Linear(hidden_size, cont_size)

        self.face_projector = nn.Linear(face_size, hidden_size) # query
        self.query_emb = nn.Embedding(3, hidden_size)
        self.txt_proj = nn.Linear(hidden_size, cont_size)
        
        self.model = VideoLocatorLlamaModel(config)

        self.span_embed = nn.Sequential(
            nn.Linear(hidden_size, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )
        self.class_embed = nn.Linear(hidden_size, 2)
        self.ground_embedding = nn.Embedding(1, hidden_size)

        self.post_init()

    def add_lora(self, lora_config):
        self.model = get_peft_model(self.model, lora_config)

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter):
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        print(self.vid_input_projector.load_state_dict(mm_projector_weights, strict=False))
        print("load mlp:", pretrain_mm_mlp_adapter)

    def get_model(self):
        return self.model

    def get_vid_outputs(self, video_emb, video_face):
        video_emb = self.vid_input_projector(video_emb)
        video_face = self.vid_face_input_projector(video_face)
        video_features = video_emb + video_face.sum(dim=2)
        vid_outputs = self.model.grounding_forward(
            inputs_embeds=video_features,
            input_type='video'
        )
        vid_emb_norm = F.normalize(self.vid_proj(vid_outputs), dim=-1)
        return vid_outputs, vid_emb_norm

    def get_txt_outputs(self, input_ids, attention_mask, query_face):
        attention_mask, inputs_embeds = self.prepare_inputs_labels_for_multimodal(
                                            input_ids,
                                            attention_mask,
                                            query_face
                                        )
        txt_outputs = self.model.grounding_forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            input_type='text'
        )
        query_pos = attention_mask.sum(dim=1) - 1
        txt_emb = torch.stack([txt_outputs[i][x] for i, x in enumerate(query_pos)])
        txt_emb_norm = F.normalize(self.txt_proj(txt_emb), dim=-1)
        return txt_outputs, txt_emb_norm, query_pos

    def grounding(self, hidden_states, query_pos, gt_label, labels, eval=False):
        # print(hidden_states.shape, query_pos)

        query_token = torch.stack([hidden_states[i][x] for i, x in enumerate(query_pos)])
        # print(query_token)
        span_out = self.span_embed(query_token)
        span = span_out.sigmoid()
        match_logit = None

        video_token = torch.stack([hidden_states[i][x-100: x] for i, x in enumerate(query_pos)])
        video_class = self.class_embed(video_token)
        if eval:
            return span, video_class, match_logit
        # print(span.shape, video_class.shape, gt.shape, gt_label.shape)

        loss_label = F.cross_entropy(video_class.view(-1, 2), gt_label.view(-1), torch.tensor([0.1, 1], device=video_class.device, dtype=video_class.dtype))

        if labels is not None:
            gt = labels
            loss_span = F.l1_loss(span, gt)
            loss_giou = (1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(span), span_cxw_to_xx(gt)))).mean()
            iou, _ = temporal_iou(span_cxw_to_xx(span), span_cxw_to_xx(gt))
            loss_match = None
            return loss_label, loss_span, loss_giou, loss_match, iou
        else:
            loss_match = F.cross_entropy(match_logit, 
                            torch.zeros(match_logit.shape[0], device=match_logit.device, dtype=torch.int64))
            return loss_label, loss_match


    def get_fusion_inputs(self, txt_outputs, vid_outputs, query_pos, indexs):
        fusion_inputs = [
            torch.cat([txt_outputs[i][:x], vid_outputs[indexs[i]][:100], self.ground_embedding.weight]) for i, x in enumerate(query_pos)
        ]
        max_len = max(x.shape[0] for x in fusion_inputs)
        fusion_inputs = [
            torch.cat([x, torch.zeros((max_len - x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)]) for x in fusion_inputs
        ]
        fusion_inputs = torch.stack(fusion_inputs)
        return fusion_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        video: Optional[torch.FloatTensor] = None,
        video_face: Optional[torch.FloatTensor] = None,
        query_face: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        gt_label: Optional[torch.Tensor] = None,
        cont_mask: Optional[torch.Tensor] = None,
    ):

        txt_outputs, txt_emb_norm, query_pos = self.get_txt_outputs(input_ids, attention_mask, query_face)
        vid_outputs, vid_emb_norm = self.get_vid_outputs(video, video_face)

        bs = input_ids.shape[0]
        pos_indexs = list(range(bs))
        fusion_inputs = self.get_fusion_inputs(txt_outputs, vid_outputs, query_pos, pos_indexs)
        fusion_outputs = self.model.grounding_forward(
            inputs_embeds=fusion_inputs,
            input_type='grounding'
        )
        fusion_query_pos = query_pos + 100
        loss_label, loss_span, loss_giou, loss_match, iou = self.grounding(fusion_outputs, fusion_query_pos, gt_label, labels)
        loss = 10 * loss_span
        loss += 4 * loss_label
        loss += 1 * loss_giou

        txt_frame_sim = torch.matmul(vid_emb_norm, txt_emb_norm.t()) / 0.07 # <bs, v_q, bs>

        sim, _ =  txt_frame_sim.max(dim=1)
        sim[cont_mask == 1] = float('-inf')
        contrastive_labels = torch.arange(sim.shape[0], device=sim.device, dtype=torch.long)
        loss_contrastive = (F.cross_entropy(sim, contrastive_labels) + F.cross_entropy(sim.t(), contrastive_labels)) / 2
        loss += 1 * loss_contrastive

        iou = torch.diag(iou)

        return {
            'loss': loss,
            'iou': iou,
            'txt': txt_emb_norm,
            'vid': vid_emb_norm
        }



    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, query_face
    ):
        if type(query_face) is list:
            concat_images = torch.cat([image for image in query_face], dim=0)
            image_features = self.face_projector(concat_images)
            split_sizes = [image.shape[0] for image in query_face]
            query_face = torch.split(image_features, split_sizes, dim=0)
        else:
            query_face = self.face_projector(query_face)

        query_emb = self.query_emb.weight
        
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]

        new_input_embeds = []
        for batch_idx, cur_input_ids in enumerate(input_ids):

            image_token_indices = [-1] + torch.where((cur_input_ids == VIDEO_TOKEN_INDEX) | (cur_input_ids == PERSON_TOKEN_INDEX))[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []

            person_id = 0
            for i in range(len(cur_input_embeds_no_im)):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i + 1 < len(cur_input_embeds_no_im):
                    raw_id = cur_input_ids[image_token_indices[i+1]]
                    if raw_id == VIDEO_TOKEN_INDEX:
                        cur_new_input_embeds.append(query_emb[:1])
                    elif raw_id == PERSON_TOKEN_INDEX:
                        cur_face_features = query_face[batch_idx][person_id: person_id+1]
                        cur_new_input_embeds.append(query_emb[1:2])
                        cur_new_input_embeds.append(cur_face_features)
                        cur_new_input_embeds.append(query_emb[2:3])
                        person_id += 1
                    else:
                        print(raw_id)
                        assert False
            assert person_id == 0 or person_id == len(query_face[batch_idx])


            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)


        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)


        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    attention_mask[i, -cur_len:] = True

            else: # <yes>
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    attention_mask[i, :cur_len] = True


        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return attention_mask, new_input_embeds

AutoConfig.register("VideoLocator", VideoLocatorConfig)
AutoModelForCausalLM.register(VideoLocatorConfig, VideoLocator)
