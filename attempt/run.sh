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
per_device_train_batch_size=32
per_device_eval_batch_size=256
load_best_model_at_end=true
metric_for_best_model="average_metrics"
greater_is_better=true
evaluation_strategy="epoch"
non_linearity="relu"
max_source_length=256
learning_rate=3e-4
output_dir="outputs/adapters"
split_validation_test=true
task_name=("wnli")
eval_dataset_name=("wnli")
test_dataset_name=("wnli")
num_train_epochs=20
dataset_config_name=("en")
eval_dataset_config_name=("en")
test_dataset_config_name=("en")
predict_with_generate=true
add_adapter_in_self_attention=false
add_layer_norm_before_adapter=true
add_layer_norm_after_adapter=false
adapter_config_name="adapter"
train_task_adapters=true
task_reduction_factor=32
unfreeze_lm_head=false
unfreeze_layer_norms=true
overwrite_output_dir=true
compute_memory=true
report_to="wandb"
add_lora=true
load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora_qnli_prefix/lora.pt"

target_task=(qnli)

for task in ${target_task[@]}
do
     
    t=($task)
    echo $t
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
        --add_adapter_in_self_attention=$add_adapter_in_self_attention \
        --add_layer_norm_before_adapter=$add_layer_norm_before_adapter \
        --add_layer_norm_after_adapter=$add_layer_norm_after_adapter \
        --adapter_config_name="$adapter_config_name" \
        --train_task_adapters=$train_task_adapters \
        --task_reduction_factor=$task_reduction_factor \
        --unfreeze_lm_head=$unfreeze_lm_head \
        --unfreeze_layer_norms=$unfreeze_layer_norms \
        --overwrite_output_dir=$overwrite_output_dir \
        --compute_memory=$compute_memory \
        --report_to="$report_to" \
        --add_lora=$add_lora \
        --load_lora_path="$load_lora_path" \
        --few_shot=16
    
    # python run_seq2seq.py \
    #     --do_train=$do_train \
    #     --do_eval=$do_eval \
    #     --do_test=$do_test \
    #     --warmup_steps=$warmup_steps \
    #     --save_steps=$save_steps \
    #     --save_strategy="$save_strategy" \
    #     --model_name_or_path="$model_name_or_path" \
    #     --tokenizer_name="$tokenizer_name" \
    #     --save_total_limit=$save_total_limit \
    #     --per_device_train_batch_size=$per_device_train_batch_size \
    #     --per_device_eval_batch_size=$per_device_eval_batch_size \
    #     --load_best_model_at_end=$load_best_model_at_end \
    #     --metric_for_best_model="$metric_for_best_model" \
    #     --greater_is_better=$greater_is_better \
    #     --evaluation_strategy="$evaluation_strategy" \
    #     --non_linearity="$non_linearity" \
    #     --max_source_length=$max_source_length \
    #     --learning_rate=$learning_rate \
    #     --output_dir="$output_dir" \
    #     --split_validation_test=$split_validation_test \
    #     --task_name="${t[@]}" \
    #     --eval_dataset_name="${t[@]}" \
    #     --test_dataset_name="${t[@]}" \
    #     --num_train_epochs=$num_train_epochs \
    #     --dataset_config_name="${dataset_config_name[@]}" \
    #     --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
    #     --test_dataset_config_name="${test_dataset_config_name[@]}" \
    #     --predict_with_generate=$predict_with_generate \
    #     --add_adapter_in_self_attention=$add_adapter_in_self_attention \
    #     --add_layer_norm_before_adapter=$add_layer_norm_before_adapter \
    #     --add_layer_norm_after_adapter=$add_layer_norm_after_adapter \
    #     --adapter_config_name="$adapter_config_name" \
    #     --train_task_adapters=$train_task_adapters \
    #     --task_reduction_factor=$task_reduction_factor \
    #     --unfreeze_lm_head=$unfreeze_lm_head \
    #     --unfreeze_layer_norms=$unfreeze_layer_norms \
    #     --overwrite_output_dir=$overwrite_output_dir \
    #     --compute_memory=$compute_memory \
    #     --report_to="$report_to" \
    #     --few_shot=16
done



# 启动Python脚本并传递配置项作为启动参数
# python run_seq2seq.py \
#     --do_train=$do_train \
#     --do_eval=$do_eval \
#     --do_test=$do_test \
#     --warmup_steps=$warmup_steps \
#     --save_steps=$save_steps \
#     --save_strategy="$save_strategy" \
#     --model_name_or_path="$model_name_or_path" \
#     --tokenizer_name="$tokenizer_name" \
#     --save_total_limit=$save_total_limit \
#     --per_device_train_batch_size=$per_device_train_batch_size \
#     --per_device_eval_batch_size=$per_device_eval_batch_size \
#     --load_best_model_at_end=$load_best_model_at_end \
#     --metric_for_best_model="$metric_for_best_model" \
#     --greater_is_better=$greater_is_better \
#     --evaluation_strategy="$evaluation_strategy" \
#     --non_linearity="$non_linearity" \
#     --max_source_length=$max_source_length \
#     --learning_rate=$learning_rate \
#     --output_dir="$output_dir" \
#     --split_validation_test=$split_validation_test \
#     --task_name="${task_name[@]}" \
#     --eval_dataset_name="${eval_dataset_name[@]}" \
#     --test_dataset_name="${test_dataset_name[@]}" \
#     --num_train_epochs=$num_train_epochs \
#     --dataset_config_name="${dataset_config_name[@]}" \
#     --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
#     --test_dataset_config_name="${test_dataset_config_name[@]}" \
#     --predict_with_generate=$predict_with_generate \
#     --add_adapter_in_self_attention=$add_adapter_in_self_attention \
#     --add_layer_norm_before_adapter=$add_layer_norm_before_adapter \
#     --add_layer_norm_after_adapter=$add_layer_norm_after_adapter \
#     --adapter_config_name="$adapter_config_name" \
#     --train_task_adapters=$train_task_adapters \
#     --task_reduction_factor=$task_reduction_factor \
#     --unfreeze_lm_head=$unfreeze_lm_head \
#     --unfreeze_layer_norms=$unfreeze_layer_norms \
#     --overwrite_output_dir=$overwrite_output_dir \
#     --compute_memory=$compute_memory \
#     --report_to="$report_to" \
#     --add_lora=$add_lora \
#     --load_lora_path="$load_lora_path"
