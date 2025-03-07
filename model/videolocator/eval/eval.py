import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
print(root_dir)
import sys
sys.path.append(root_dir)

from videolocator.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import torch
from transformers import AutoTokenizer
from videolocator.model.builder import load_lora
from videolocator.train.dataset import GroundingDataset, DataCollatorForGroundingDataset
from videolocator.model import VideoLocator
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Mapping
import numpy as np
import json
from peft import LoraConfig

def prepare_input(data):
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        if data.dtype in [torch.float64, torch.float32, torch.float16]:
            data = data.to(torch.bfloat16)
        return data.to('cuda')
    

def cal_grounding_r(x, prefix, total):
    ret = {
        prefix + str(m): float(sum(x >= m)) / total
            for m in [0.3, 0.5, 0.7]
    }
    return ret

def eval_metrics(model, data_loader):
    txt_emb = []
    vid_emb = []
    ious = []
    retrieval_labels = np.array(data_loader.dataset.retrieval_labels)
    uni_ids = data_loader.dataset.uni_ids

    for data in tqdm(data_loader):
        with torch.inference_mode():
            data = prepare_input(data)
            txt_outputs, t_emb, query_pos = model.get_txt_outputs(data['input_ids'], data['attention_mask'], data['query_face'])
            vid_outputs, v_emb = model.get_vid_outputs(data['video'], data['video_face'])

            fusion_input = model.get_fusion_inputs(txt_outputs, vid_outputs, query_pos, list(range(query_pos.shape[0])))
            fusion_outputs = model.model.grounding_forward(
                    inputs_embeds=fusion_input,
                    input_type='grounding'
                )
            fusion_query_pos = query_pos + 100
            _, _, _, _, iou = model.grounding(fusion_outputs, fusion_query_pos, data['gt_label'], data['labels'])

            txt_emb.append(t_emb)
            vid_emb.append(v_emb)
            ious.append(torch.diag(iou).cpu())

    txt_emb = torch.cat(txt_emb)
    vid_emb = torch.cat(vid_emb)[uni_ids, :, :]
    ious = torch.cat(ious)
    metrics = {}

    sim = torch.matmul(vid_emb, txt_emb.t()).max(dim=1)[0].t()
    first_true = []
    n = sim.shape[0]
    VCG_iou = []

    sim_rank = sim.argsort(dim=1, descending=True)
    correct_ids = []
    for i in range(n):
        for j in range(n):
            if sim_rank[i][j] == retrieval_labels[i]:
                first_true.append(j)
                if j == 0:
                    VCG_iou.append(ious[i])
                    correct_ids.append(i)
                break
    assert len(first_true) == n

    metrics.update({
        'VR_r' + str(m): float(sum([x < m for x in first_true])) / n
            for m in [1, 5, 10, 25, 50, 100, 500]
    })
    metrics.update(cal_grounding_r(ious, 'SVCG_R@', len(ious)))
    metrics.update(cal_grounding_r(torch.tensor(VCG_iou), 'VCG_R@', len(ious)))
    print(metrics)
    return metrics
    

def eval(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = VideoLocator.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, **{'torch_dtype': torch.bfloat16})
    model.initialize_vision_modules(args.pretrain_mm_mlp_adapter)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=['gate_proj', 'up_proj', 'v_proj', 'o_proj', 'q_proj', 'k_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type="CAUSAL_LM",
    )
    model.add_lora(lora_config)
    lora_v = torch.load(os.path.join(args.lora_path, "pytorch_model.bin"), map_location='cpu')
    model.load_state_dict(lora_v, strict=False)
    non_lora_trainables = torch.load(os.path.join(args.lora_path, 'non_lora_trainables.bin'), map_location='cpu')
    model.load_state_dict(non_lora_trainables, strict=False)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()

    metrics = {}

    ds = {'seen': GroundingDataset(tokenizer, args, 'test_seen.jsonl'),
          'unseen': GroundingDataset(tokenizer, args, 'test_unseen.jsonl'),}

    for k, dataset in ds.items():
        data_collator = DataCollatorForGroundingDataset(tokenizer)
        data_loader = DataLoader(dataset, batch_size=args.bs, collate_fn=data_collator, num_workers=4, shuffle=False)
        metrics[k] = eval_metrics(model, data_loader)

    # test_json = os.path.join(lora_path, 'test.json')
    # with open(test_json, 'w') as f:
    #     json.dump(metrics, f, indent=4)
    print(metrics)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="../../checkpoints/visual_adapter/mm_projector.bin")
    parser.add_argument("--lora_path", type=str, default="../../checkpoints/stage2-test")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--data_folder", type=str, default='../../../data/annotations')
    parser.add_argument("--feat_folder", type=str, default='../../../data/feat')
    parser.add_argument("--use_face", default=True)
    parser.add_argument("--bs", type=int, default=96)
    args = parser.parse_args()

    # disable_torch_init()

    eval(args)



