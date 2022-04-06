cd data/multiwoz
unzip total.json.zip
unzip val.json.zip
cd ../../  # (working directory)

wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin -P convlab/modules/e2e/multiwoz/Transformer/pytorch_transformers/

python -m torch.distributed.launch --nproc_per_node=4 convlab/modules/e2e/multiwoz/Transformer/train.py \
     --dataset_path=./data/multiwoz/ --dataset_cache=dataset_cache_np_gpt_total --model_checkpoint=gpt2 \
     --model_version=v5 --lr=5e-4 --train_batch_size=2 --valid_batch_size=1  \
     --max_history=15 --gradient_accumulation_steps=4 --n_epochs=15  --adapter=1 \
     --n_adapter=512 --log_dir=./runs/ConvLab_1_31
