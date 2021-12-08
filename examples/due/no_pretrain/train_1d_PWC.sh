#!/usr/bin/env bash
set -x
shopt -s extglob
PYTHONPATH="." CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 benchmarker/cli/l5/train.py \
--model_name_or_path ${INIT_LARGE_MODEL_PATH} \
--relative_bias_args='[{"type":"1d"}]' \
--dropout_rate 0.15 \
--label_smoothing 0.15 \
--model_type=t5  \
--data_dir ${DATASETS_ROOT}/PWC/${OCR_ENGINE}/train \
--val_data_dir ${DATASETS_ROOT}/PWC/${OCR_ENGINE}/dev \
--test_data_dir  ${DATASETS_ROOT}/PWC/${OCR_ENGINE}/test \
--gpus 8 \
--num_workers 8 \
--train_batch_size 1 \
--eval_batch_size 1 \
--accumulate_grad_batches 8 \
--max_epochs 30 \
--val_check_interval 1.0 \
--output_dir ${OUT_DIR}
--max_target_length 64 \
--eval_max_gen_length 64 \
--warmup_steps 400 \
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
--seed 4 \
--accelerator=ddp \
;

