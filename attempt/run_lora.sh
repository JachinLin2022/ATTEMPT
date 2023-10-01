#!/bin/bash

# 配置项作为变量
do_train=true
do_eval=true
do_test=true
warmup_steps=500
save_steps=1000
save_strategy="no"
model_name_or_path="t5-base"
tokenizer_name="t5-base"
save_total_limit=1
per_device_train_batch_size=128
per_device_eval_batch_size=256
load_best_model_at_end=true
metric_for_best_model="average_metrics"
greater_is_better=true
evaluation_strategy="epoch"
non_linearity="gelu_new"
max_source_length=256
learning_rate=5e-4
split_validation_test=true
task_name=("wnli")
eval_dataset_name=("wnli")
test_dataset_name=("wnli")
num_train_epochs=20
dataset_config_name=("en")
eval_dataset_config_name=("en")
test_dataset_config_name=("en")
predict_with_generate=true
overwrite_output_dir=true
compute_memory=true
report_to="wandb"
train_lora=true
add_lora=true
target_task=(mrpc cola sst2 qnli wnli mnli qqp stsb superglue-boolq superglue-rte superglue-cb superglue-copa superglue-multirc superglue-wic superglue-wsc.fixed superglue-record)
target_task=(qqp mnli qnli sst2 superglue-record)
big_task=(mnli qnli qqp sst2 superglue-record)
for task in ${target_task[@]}
do
    bash clean.sh
    output_dir="outputs/lora2/lora_"$task
    t=($task)
    num_train_epochs=20
    if [[ "${big_task[@]}" =~ "${task}" ]]; then
        num_train_epochs=5
    fi
    echo $task $num_train_epochs
        python run_seq2seq.py \
        --do_train=$do_train \
        --do_eval=$do_eval \
        --do_test=$do_test \
        --warmup_steps=$warmup_steps \
        --save_steps=$save_steps \
        --save_strategy="$save_strategy" \
        --model_name_or_path="$model_name_or_path" \
        --tokenizer_name="$tokenizer_name" \
        --save_total_limit=$save_total_limit \
        --per_device_train_batch_size=$per_device_train_batch_size \
        --per_device_eval_batch_size=$per_device_eval_batch_size \
        --load_best_model_at_end=$load_best_model_at_end \
        --metric_for_best_model="$metric_for_best_model" \
        --greater_is_better=$greater_is_better \
        --evaluation_strategy="$evaluation_strategy" \
        --non_linearity="$non_linearity" \
        --max_source_length=$max_source_length \
        --learning_rate=$learning_rate \
        --output_dir="$output_dir" \
        --split_validation_test=$split_validation_test \
        --task_name="${t[@]}" \
        --eval_dataset_name="${t[@]}" \
        --test_dataset_name="${t[@]}" \
        --num_train_epochs=$num_train_epochs \
        --dataset_config_name="${dataset_config_name[@]}" \
        --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
        --test_dataset_config_name="${test_dataset_config_name[@]}" \
        --predict_with_generate=$predict_with_generate \
        --overwrite_output_dir=$overwrite_output_dir \
        --compute_memory=$compute_memory \
        --report_to="$report_to" \
        --train_lora=$train_lora \
        --add_lora=$add_lora 
done

