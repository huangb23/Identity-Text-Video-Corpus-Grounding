import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from videolocator.model import *
from peft import PeftModel

def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model

def load_pretrained_model(args, stage2=None):
    kwargs = {'torch_dtype': torch.float16}

    # model_path = os.path.expanduser(args.model_path)
    model_base = args.model_base


    # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    print('Loading VideoLocator from base model...')

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    model = VideoLocatorLlamaForGrounding.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    model.get_model().initialize_vision_modules(args)

    if stage2 is not None:
        print('Loading stage2 weights...')
        model = load_lora(model, stage2)
        print('Merging stage2 weights...')
        model = model.merge_and_unload()

    return tokenizer, model
