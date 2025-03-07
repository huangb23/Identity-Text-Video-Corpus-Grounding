```bash
git clone https://github.com/huangb23/Identity-Text-Video-Corpus-Grounding.git
cd Identity-Text-Video-Corpus-Grounding/model
pip install -r requirements.txt
```

Before training, you need to prepare the features following the instructions in [data/feats/README.md](../data/feats/README.md).

```bash
bash scripts/train.sh
bash scripts/eval.sh
```

We recommend using a GPU with at least 30GB of memory for training. Otherwise, you can reduce the batch size, but this may affect contrastive learning, leading to a decline in the model's retrieval performance. (The model in the paper was trained with batch_size=96 using DeepSpeed ZeRO-3 optimization.)