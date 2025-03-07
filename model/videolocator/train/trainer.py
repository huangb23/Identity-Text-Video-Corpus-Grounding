import os
import torch
import json
from transformers import Trainer
from typing import Optional

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, denumpify_detensorize
import numpy as np
from tqdm import tqdm

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return



def cal_r(t2v_sim, re_label, prefix):
    sim_rank = t2v_sim.argsort(dim=1, descending=True)
    first_true = []
    n = sim_rank.shape[0]
    for i in range(n):
        for j in range(n):
            if sim_rank[i][j] == re_label[i]:
                first_true.append(j)
                break
    assert len(first_true) == n
    ret = {
        prefix + str(m): float(sum([x < m for x in first_true])) / n
            for m in [1, 5, 10, 25, 50, 100, 500]
    }
    return ret

def cal_grounding_r(x, prefix):
    ret = {
        prefix + str(m): float(sum(x >= m)) / len(x)
            for m in [0.3, 0.5, 0.7]
    }
    return ret
    

class VideoLocatorTrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):

        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Only save Adapter
        keys_to_match = ['mm_projector', 'span_embed', 'class_embed', 'ground_embedding']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
    #     if getattr(self.args, 'tune_mm_mlp_adapter', False):
    #         pass
    #     else:
    #         super(VideoLocatorTrainer, self)._save(output_dir, state_dict)
            

    # def _my_eval(self, vid_emb, txt_emb, retrieval_labels, iou):
    #     NT = 4157
    #     NV = 1257
    #     out_t2v_sim = torch.matmul(vid_emb[:NV], txt_emb[:NT].t()).max(dim=1)[0].t()
    #     out_label = retrieval_labels[:NT]
    #     in_t2v_sim = torch.matmul(vid_emb[NV:], txt_emb[NT:].t()).max(dim=1)[0].t()
    #     in_label = retrieval_labels[NT:] - NV

    #     metrics = {}
    #     metrics.update(cal_grounding_r(iou[:NT], 'R@'))
    #     metrics.update(cal_grounding_r(iou[NT:], 'inR@'))
    #     metrics.update(cal_r(out_t2v_sim, out_label, 'r'))
    #     metrics.update(cal_r(in_t2v_sim, in_label, 'in_r'))
    #     return metrics

    def _my_eval(self, vid_emb, txt_emb, retrieval_labels, iou):

        out_t2v_sim = torch.matmul(vid_emb, txt_emb.t()).max(dim=1)[0].t()
        out_label = retrieval_labels

        metrics = {}
        metrics.update(cal_grounding_r(iou, 'R@'))
        metrics.update(cal_r(out_t2v_sim, out_label, 'r'))
        return metrics

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)


        # logger.info(f"***** Running {description} *****")
        # if has_length(dataloader):
        #     logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        # else:
        #     logger.info("  Num examples: Unknown")
        # logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        iou = []
        txt_emb = []
        vid_emb = []
        retrieval_labels = np.array(dataloader.dataset.retrieval_labels)
        uni_ids = dataloader.dataset.uni_ids

        for inputs in dataloader:
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            iou.append(logits[0])
            txt_emb.append(logits[1].cpu())
            vid_emb.append(logits[2].cpu())
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        iou = torch.cat(iou).cpu()
        txt_emb = torch.cat(txt_emb) # <1w+, 768>
        # vf2t_sim = [torch.matmul(v, txt_emb.t()) for v in tqdm(vid_emb)]
        # vf2t_sim = torch.cat(vf2t_sim) # N, qv, N
        vid_emb = torch.cat(vid_emb)[uni_ids, :, :] # <3183, 100, 768>


        metrics = self._my_eval(vid_emb, txt_emb, retrieval_labels, iou)

        with open(os.path.join(self.args.output_dir, 'eval.jsonl'), 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        num_samples = len(eval_dataset)


        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)
    
class VideoLocatorPreTrainer(VideoLocatorTrainer):

    def _my_eval(self, vid_emb, txt_emb, retrieval_labels, iou):
        NT = 4157
        NV = 1257
        out_t2v_sim = torch.matmul(vid_emb[:NV], txt_emb[:NT].t()).max(dim=1)[0].t()
        out_label = retrieval_labels[:NT]
        in_t2v_sim = torch.matmul(vid_emb[NV:], txt_emb[NT:].t()).max(dim=1)[0].t()
        in_label = retrieval_labels[NT:] - NV

        metrics = {}
        metrics.update(cal_r(out_t2v_sim, out_label, 'r'))
        metrics.update(cal_r(in_t2v_sim, in_label, 'in_r'))
        return metrics