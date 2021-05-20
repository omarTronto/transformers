#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

export BS=4
export GAS=1

python finetune.py \
    --learning_rate=5e-5 \
    --gpus 0 \
    --do_train \
    --do_predict \
    --val_check_interval 0.25 \
    --n_val -1 \
    --num_train_epochs 3 \
    --freeze_encoder --freeze_embeds --data_dir emails \
    --max_target_length 142 --val_max_target_length=142 \
    --train_batch_size=$BS --eval_batch_size=$BS --gradient_accumulation_steps=$GAS \
    --model_name_or_path dbart_emails_12_6 \
    --tokenizer_name facebook/bart-large \
    --warmup_steps 0 \
    --output_dir distilbart-emails-12-3-4 \
    "$@"