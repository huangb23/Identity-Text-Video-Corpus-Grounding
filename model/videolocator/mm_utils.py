import numpy as np
import torch
from videolocator.constants import VIDEO_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN, PERSON_TOKEN_INDEX
import re
import random

def tokenizer_image_token(prompt, tokenizer, image_token_index=VIDEO_TOKEN_INDEX, return_tensors=None):
    tags = [DEFAULT_VIDEO_TOKEN, '<person>']
    pattern = re.compile('|'.join([re.escape(tag) for tag in tags]))
    matches = re.finditer(pattern, prompt)
    tags_content = [match.group() for match in matches]
    splitted_text = re.split(pattern, prompt)
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in splitted_text]
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    assert len(prompt_chunks) == len(tags_content) + 1
    for i, fragment in enumerate(prompt_chunks):
        input_ids.extend(fragment[offset:])
        if i < len(tags_content):
            if tags_content[i] == DEFAULT_VIDEO_TOKEN:
                input_ids.append(VIDEO_TOKEN_INDEX)
            elif tags_content[i] == '<person>':
                input_ids.append(PERSON_TOKEN_INDEX)
            else:
                assert False
 
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        print(_, param.requires_grad, param.numel())
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False