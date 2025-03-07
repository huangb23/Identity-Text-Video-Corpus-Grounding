import random
import copy
import json
import torch
import os
import transformers
import h5py
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from videolocator.mm_utils import tokenizer_image_token

@dataclass
class DataArguments:
    data_folder: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    feat_folder: Optional[str] = field(default=None)
    use_face: bool = field(default=True)


def norm(x):
    l = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (l + 1e-6)


class GroundingDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config, split_name):
        super(GroundingDataset, self).__init__()

        self.tokenizer = tokenizer
        self.config = config
        data_path = os.path.join(config.data_folder, split_name)
        with open(data_path, 'r') as f:
            self.list_data_dict = [json.loads(x) for x in f.readlines()]

        self.video_feat = h5py.File(os.path.join(config.feat_folder, 'video_clip_feat.h5'))
        if self.config.use_face:
            self.face_feat = h5py.File(os.path.join(config.feat_folder, 'face_feat.h5'))
            self.video_face_feat = h5py.File(os.path.join(config.feat_folder, 'video_face_feat.h5'))
        
        self.cal_uni_vid()

    def cal_uni_vid(self):
        m = {}
        self.retrieval_labels = []
        self.uni_ids = []
        for i, x in enumerate(self.list_data_dict):
            v = x['vid_name']
            if v not in m:
                m[v] = len(self.uni_ids)
                self.uni_ids.append(i)
            self.retrieval_labels.append(m[v])
        self.uni_ids = np.array(self.uni_ids)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source = self.list_data_dict[i]
        id = source['vid_name']
        sentence = source['desc']
        video = self.video_feat[id][:]
        sample_indices = np.linspace(0, len(video) - 1, 100, dtype=int)
        video = video[sample_indices]

        if self.config.use_face:
            video_face = self.video_face_feat[id][:]
            video_face = video_face[sample_indices]
            video_face = norm(video_face)

            show = id.split("_")[0]
            if show.startswith("s"):
                show = 'bbt'
            query_face = []

            sentence_part = sentence.split('<person>')
            sentence = sentence_part[0]
            for person_i, name in enumerate(source["person"]):
                key = f"{show}_{name}"
                if key in self.face_feat.keys():
                    query_face.append(self.face_feat[key])
                    sentence += '<person>'
                else:
                    sentence += name
                sentence += sentence_part[person_i+1]
            if len(query_face) > 0:
                query_face = np.array(query_face)
                query_face = norm(query_face)
            else:
                query_face = np.zeros((1, 512))      
        else:
            video_face = np.zeros((100, 3, 512))    
            query_face = np.zeros((1, 512))      

        prompt = sentence + '<video>'
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt')
        if len(input_ids) > 70:
            print(len(input_ids), sentence)
            return random.choice(self)

        se = [x / source['duration'] for x in source['ts']]
        gt = torch.tensor([(se[1] + se[0]) / 2, se[1] - se[0]])
        gt_label = torch.tensor([se[0] <= x/100 <= se[1] for x in range(100)], dtype=torch.int64)

        data_dict = dict(id=id,
                         input_ids=input_ids,
                         video=torch.from_numpy(video),
                         video_face=torch.from_numpy(video_face),
                         query_face=torch.from_numpy(query_face),
                         gt=gt,
                         gt_label=gt_label)
        return data_dict
    


@dataclass
class DataCollatorForGroundingDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        # print(input_ids.shape)

        ids = [instance["id"] for instance in instances]

        n = len(ids)
        cont_mask = torch.zeros((n, n), dtype=torch.int)
        for i in range(n):
            for j in range(n):
                if i != j and ids[i] == ids[j]:
                    cont_mask[i][j] = 1

        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            video=torch.stack([instance["video"] for instance in instances]),
            video_face=torch.stack([instance["video_face"] for instance in instances]),
            labels=torch.stack([instance["gt"] for instance in instances]),
            gt_label=torch.stack([instance["gt_label"] for instance in instances]),
            query_face=[instance["query_face"] for instance in instances],
            cont_mask = cont_mask
        )

        return batch



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = GroundingDataset(tokenizer, data_args, "train.jsonl")
    eval_dataset = GroundingDataset(tokenizer, data_args, 'val_seen.jsonl')
    data_collator = DataCollatorForGroundingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

