import torch
import os

from transformers import AutoModelForCausalLM

import argparse

from datautils import get_loaders
from hook import get_act_stats_llama


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='meta-llama/Llama-2-7b-hf', help='model name')
    parser.add_argument('--output-path', type=str, default='act_scales/llama2-7b-hf.pt',
                        help='where to save the act scales')
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--seqlen', type=int, default=2048)

    # microxscaling settings
    parser.add_argument("--mx", action="store_true", help="Whether to use microxcaling")
    parser.add_argument("--mx_format", type=str, choices=["int8", "int4", "fp8_e5m2", "fp8_e4m3",
                            "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"], help="MX element format")
    parser.add_argument("--mx_block_size", type=int, default=32, help="MX block size")

    # device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="device to use")

    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                 torch_dtype=torch.float32,
                                                 device_map="cpu")
    model.seqlen = args.seqlen
    dataloader, testloader = get_loaders(
        "wikitext2", nsamples=args.nsamples, seed=42, model=args.model_name, seqlen=model.seqlen
    )

    if (args.mx):
        print("Using microxscaling dataformat.")
        from mx import mx_mapping
        from mx import finalize_mx_specs
        mx_specs = {
            'w_elem_format': args.mx_format, #'int8',#'fp6_e3m2',
            'a_elem_format': args.mx_format, #'int8',#'fp6_e3m2',
            'block_size': args.mx_block_size, #32,
            'bfloat': 16,
            'custom_cuda': True,
            'quantize_backprop': False,
        }
        mx_specs = finalize_mx_specs(mx_specs)
        mx_mapping.inject_pyt_ops(mx_specs)


    act_scales = get_act_stats_llama(model, dataloader, args.device)


    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()
