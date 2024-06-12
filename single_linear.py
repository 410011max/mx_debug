import torch
import random
import numpy as np
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    
    # microxscaling settings
    parser.add_argument("--mx", action="store_true", help="Whether to use microxcaling")
    parser.add_argument("--mx_format", type=str, choices=["int8", "int4", "fp8_e5m2", "fp8_e4m3",
                            "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"], help="MX element format")
    parser.add_argument("--mx_block_size", type=int, default=32, help="MX block size")

    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

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


    linear = torch.nn.Linear(32, 1)
    input = torch.randn(32)

    output = linear(input)

    print(f'Linear: {linear.weight}')
    print(f'Input: {input}')
    print(f'Output: {output}')


if __name__ == '__main__':
    main()    
