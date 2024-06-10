import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from calibration import get_act_acales


def build_model_and_tokenizer(model_name, mx, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=2048)
    kwargs = {"torch_dtype": torch.float32 if mx else torch.float16, "device_map": device}
    print(f"Using {kwargs['torch_dtype']} on {kwargs['device_map']}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='meta-llama/Llama-2-7b-hf', help='model name')
    parser.add_argument('--output-path', type=str, default='act_scales/llama2-7b-hf.pt',
                        help='where to save the act scales')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=2048)

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
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.mx, args.device)

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


    act_scales = get_act_acales(model, tokenizer, args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()
