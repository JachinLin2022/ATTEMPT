#!/bin/bash

# 配置项作为变量
do_train=true
do_eval=true
do_test=true
warmup_steps=0
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
dataset_config_name=("en")
eval_dataset_config_name=("en")
test_dataset_config_name=("en")
predict_with_generate=true
overwrite_output_dir=true
compute_memory=true
report_to="none"
add_lora=true
train_lora=true
add_task_embedding=true

big_task=(superglue-multirc)
small_task=(superglue-cb superglue-wsc-fixed)
lrs=(3e-4 6e-4 1e-3 5e-3)


task_reduction_factor=16
target_task=(superglue-cb superglue-wsc-fixed cola rte mrpc stsb superglue-wic superglue-boolq superglue-multirc)
target_task=(superglue-cb)
for task in ${target_task[@]}
do
    for learning_rate in ${lrs[@]}
    do
        t=($task)

        num_train_epochs=20
        warmup_steps=0
        per_device_train_batch_size=128
        max_source_length=256
        if [[ "${big_task[@]}" =~ "${task}" ]]; then
            num_train_epochs=10
            max_source_length=348
        fi

        if [[ "${small_task[@]}" =~ "${task}" ]]; then
            per_device_train_batch_size=32
        fi

        load_adapter_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/mnli/adapter.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/qnli/adapter.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/qqp/adapter.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/sst2/adapter.pt"
        load_task_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/mnli/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/qnli/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/qqp/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage1/sst2/task_embedding.pt"
        output_dir="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/adapter/stage2/"$learning_rate"_"$task"_"$per_device_train_batch_size"_"$task_reduction_factor

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
            --add_task_embedding=$add_task_embedding \
            --unfreeze_lm_head=false \
            --unfreeze_layer_norms=true \
            --add_adapter_in_feed_forward=true \
            --add_adapter_in_feed_forward_out=true \
            --add_adapter_in_self_attention=false \
            --add_layer_norm_before_adapter=false \
            --add_layer_norm_after_adapter=false \
            --adapter_config_name="adapter" \
            --train_task_adapters=true \
            --task_reduction_factor=$task_reduction_factor \
            --load_adapter_path=$load_adapter_path \
            --load_task_path=$load_task_path \
            --logging_steps 10
        bash clean.sh $output_dir
    done
done