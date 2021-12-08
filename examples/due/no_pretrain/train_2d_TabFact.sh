#!/usr/bin/env bash
set -x
shopt -s extglob
PYTHONPATH="." CUDA_VISIBLE_DEVICES=5,6,7,8 benchmarker/cli/l5/train.py \
--model_name_or_path /data-c/shared/athena/models/hf/t5-base/ \
--relative_bias_args='[{"type":"1d"},{"type":"horizontal"},{"type":"vertical"}]' \
--dropout_rate 0.15 \
--label_smoothing 0 \
--model_type=t5  \
--data_dir ${DATASETS_ROOT}/TabFact/${OCR_ENGINE}/train \
--val_data_dir ${DATASETS_ROOT}/TabFact/${OCR_ENGINE}/dev \
--test_data_dir  ${DATASETS_ROOT}/TabFact/${OCR_ENGINE}/test \
--gpus 4 \
--num_workers 16 \
--train_batch_size 16 \
--eval_batch_size 16 \
--accumulate_grad_batches 1 \
--max_epochs 30 \
--val_check_interval 0.20 \
--output_dir ${OUT_DIR}
--max_target_length 256 \
--eval_max_gen_length 256 \
--warmup_steps 100 \
--learning_rate 2e-4  \
--lr_scheduler constant \
--val_metric val_loss \
--do_train \
--do_predict \
--additional_data_fields doc_id label_name \
--segment_levels tokens pages \
--optimizer adamw \
--weight_decay 1e-5 \
--adam_epsilon 1e-8 \
--gradient_checkpointing \
--trim_batches \
--accelerator=ddp \
--seed 4 \
--max_source_length 1024 \
--early_stopping_patience 20 \
;


