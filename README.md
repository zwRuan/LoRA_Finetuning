# LoRA_Finetuning

## Env
```
conda create -n lora_finetuning python=3.10 -y
conda activate lora_finetuning

# if use gpu and cuda verison is 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# if use cpu
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# then
pip install -r requirements.txt
```

## download model
```
# modelscope
modelscope download --model Qwen/Qwen3-0.6B

# or huggingface
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen3-0.6B
```


## finetuning qwen3-0.6B
```
# if use gpu
bash lora_gpu.sh
# if use cpu
bash lora_cpu
```

## merge model
```
python merge.py --base_model "Qwen/Qwen3-0.6B" --adapter output/qwen3_lora_adapter --output_path merge/qwen3_lora
```

## generate
```
python generate.py
```