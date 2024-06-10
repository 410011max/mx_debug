# original
CUDA_VISIBLE_DEVICES=0 \
python generate_act_scales.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-path act_scales/llama2-7b-hf.pt \
    --nsamples 1 \
    --seqlen 1 \
    --device cuda

# mx version
CUDA_VISIBLE_DEVICES=0 \
python generate_act_scales.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-path act_scales/llama2-7b-hf-mx.pt \
    --nsamples 1 \
    --seqlen 1 \
    --device cpu \
    --mx --mx_format fp4_e2m1 --mx_block_size 32
    