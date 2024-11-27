torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    ./model/main_pretrain.py \
        --epoch 2000 \
        --batch_size 32 \
        --tile_size 128 \
        --output_dir "/gpfs/scratch/jt3545/projects/CODEX/analysis/lung_imc/model_ckpt_a100" \
        --log_dir "/gpfs/scratch/jt3545/projects/CODEX/analysis/lung_imc/model_ckpt_100/logs" \
        --data_path "/gpfs/scratch/jt3545/projects/CODEX/data_processed/lung_imc/data" \
