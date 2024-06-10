import torch
import torch.nn as nn

from datasets import load_dataset
from collections import defaultdict

import numpy as np
from functools import partial
from tqdm import tqdm


@torch.no_grad()
def get_act_acales(model, tokenizer, num_samples=128, seq_len=2048):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]

        # save all tensor x
        x = x.clone().view(-1, x.shape[-1]).abs().detach().cpu()
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x
        else:
            act_dict[name]["input"] = torch.concat((act_dict[name]["input"], x), dim=0)
        
        if isinstance(y, tuple):
            y = y[0]
        
        # save all tensor y
        y = y.clone().view(-1, y.shape[-1]).abs().detach().cpu()
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y
        else:
            act_dict[name]["output"] = torch.concat((act_dict[name]["output"], y), dim=0)

        del x, y

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            # format: "model.layers.0.self_attn.q_proj" or "model.layers.0.mlp.gate_proj"

            # only save the first layer
            if "0" in name:
                hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()


    return act_dict
