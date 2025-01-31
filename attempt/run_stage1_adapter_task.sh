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
dataset_config_name=("en")
eval_dataset_config_name=("en")
test_dataset_config_name=("en")
predict_with_generate=true
overwrite_output_dir=true
compute_memory=true
report_to="none"
add_task_embedding=true
init_task_from_vocab=true
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export http_proxy='http://127.0.0.1:7890'
export https_proxy='http://127.0.0.1:7890'
task_reduction_factor=48

target_task=(superglue-record squad)
for task in ${target_task[@]}
do
    t=($task)
    num_train_epochs=5
    output_dir="/home/linzhisheng/ATTEMPT/attempt/result/adapter/stage1/"$task
    per_device_train_batch_size=128
    per_device_eval_batch_size=256
    max_source_length=256
    if [[ "superglue-record" =~ "${task}" ]] || [[ "squad" =~ "${task}" ]]; then
        echo 123123123
        per_device_train_batch_size=64
        per_device_eval_batch_size=128
        max_source_length=512
    fi

    echo $task $num_train_epochs
    while true
    do
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
        --init_task_from_vocab=$init_task_from_vocab \
        --unfreeze_lm_head=false \
        --unfreeze_layer_norms=false \
        --add_adapter_in_feed_forward=false \
        --add_adapter_in_feed_forward_out=true \
        --add_adapter_in_self_attention=false \
        --add_layer_norm_before_adapter=false \
        --add_layer_norm_after_adapter=false \
        --adapter_config_name="adapter" \
        --train_task_adapters=true \
        --task_reduction_factor=$task_reduction_factor \
        --logging_steps 10
        if [[ $? -eq 0 ]]; then
            # 如果脚本正常退出，则跳出循环
            break
        fi
    done
    bash clean.sh $output_dir
done



export CUDA_VISIBLE_DEVICES="0"
target_task=(qnli mnli qqp sst2)
for task in ${target_task[@]}
do
    t=($task)
    num_train_epochs=5
    output_dir="/home/linzhisheng/ATTEMPT/attempt/result/adapter/stage1/"$task
    per_device_train_batch_size=128
    per_device_eval_batch_size=256
    max_source_length=256
    if [[ "superglue-record" =~ "${task}" ]] || [[ "squad" =~ "${task}" ]]; then
        echo 123123123
        per_device_train_batch_size=64
        per_device_eval_batch_size=128
        max_source_length=512
    fi

    echo $task $num_train_epochs
    while true
    do
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
        --init_task_from_vocab=$init_task_from_vocab \
        --unfreeze_lm_head=false \
        --unfreeze_layer_norms=false \
        --add_adapter_in_feed_forward=false \
        --add_adapter_in_feed_forward_out=true \
        --add_adapter_in_self_attention=false \
        --add_layer_norm_before_adapter=false \
        --add_layer_norm_after_adapter=false \
        --adapter_config_name="adapter" \
        --train_task_adapters=true \
        --task_reduction_factor=$task_reduction_factor \
        --logging_steps 10
        if [[ $? -eq 0 ]]; then
            # 如果脚本正常退出，则跳出循环
            break
        fi
    done
    bash clean.sh $output_dir
done

export CUDA_VISIBLE_DEVICES="0,1,2,3"
target_task=(qnli mnli qqp sst2)
for task in ${target_task[@]}
do
    t=($task)
    num_train_epochs=5
    output_dir="/home/linzhisheng/ATTEMPT/attempt/result/adapter/stage1/"$task"_4gpu"
    per_device_train_batch_size=128
    per_device_eval_batch_size=256
    max_source_length=256
    if [[ "superglue-record" =~ "${task}" ]] || [[ "squad" =~ "${task}" ]]; then
        echo 123123123
        per_device_train_batch_size=64
        per_device_eval_batch_size=128
        max_source_length=512
    fi

    echo $task $num_train_epochs
    while true
    do
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
        --init_task_from_vocab=$init_task_from_vocab \
        --unfreeze_lm_head=false \
        --unfreeze_layer_norms=false \
        --add_adapter_in_feed_forward=false \
        --add_adapter_in_feed_forward_out=true \
        --add_adapter_in_self_attention=false \
        --add_layer_norm_before_adapter=false \
        --add_layer_norm_after_adapter=false \
        --adapter_config_name="adapter" \
        --train_task_adapters=true \
        --task_reduction_factor=$task_reduction_factor \
        --logging_steps 10
        if [[ $? -eq 0 ]]; then
            # 如果脚本正常退出，则跳出循环
            break
        fi
    done
    bash clean.sh $output_dir
done