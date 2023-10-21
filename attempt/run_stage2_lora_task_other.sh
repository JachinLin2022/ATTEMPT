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
per_device_eval_batch_size=128
load_best_model_at_end=true
metric_for_best_model="average_metrics"
greater_is_better=true
evaluation_strategy="epoch"
non_linearity="gelu_new"
max_source_length=256
learning_rate=3e-4
split_validation_test=true
dataset_config_name=("en")
eval_dataset_config_name=("en")
test_dataset_config_name=("en")
predict_with_generate=true
overwrite_output_dir=true
compute_memory=true
report_to="none"
add_lora=true




big_task=(superglue-multirc)
small_task=(superglue-cb superglue-wsc-fixed)

task_reduction_factor=16

target_task=(scitail paws winogrande)


lrs=(3e-4 6e-4 1e-3)
target_task=(scitail paws winogrande yelp_polarity)
for learning_rate in ${lrs[@]}
do
    for task in ${target_task[@]}
    do
        t=($task)
        num_train_epochs=20
        warmup_steps=0
        per_device_train_batch_size=64
        max_source_length=256
        if [[ "${big_task[@]}" =~ "${task}" ]]; then
            num_train_epochs=10
            max_source_length=348
        fi

        if [[ "${small_task[@]}" =~ "${task}" ]]; then
            per_device_train_batch_size=32
        fi

        load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/mnli_fp32/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qnli_fp32/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qqp_fp32/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/sst2_fp32/lora.pt"
        load_task_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/mnli_fp32/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qnli_fp32/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qqp_fp32/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/sst2_fp32/task_embedding.pt"
        output_dir="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage2_random_init/"$learning_rate"_"$task"_"$per_device_train_batch_size"_"$task_reduction_factor"_20"

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
            --adapter_config_name="adapter" \
            --train_task_adapters=true \
            --task_reduction_factor=$task_reduction_factor \
            --unfreeze_lm_head=false \
            --unfreeze_layer_norms=true \
            --add_adapter_in_feed_forward=true \
            --add_adapter_in_feed_forward_out=false \
            --add_adapter_in_self_attention=false \
            --add_layer_norm_before_adapter=false \
            --add_layer_norm_after_adapter=false \
            --add_lora=$add_lora \
            --add_task_embedding=true \
            --load_task_path=$load_task_path \
            --init_task_from_vocab=false \
            --logging_steps 10 \
            --load_lora_path=$load_lora_path
        bash clean.sh $output_dir
    done
done

# target_task=(superglue-wic)
# task_reduction_factor=32
# for learning_rate in ${lrs[@]}
# do
#     for task in ${target_task[@]}
#     do
#         t=($task)
#         num_train_epochs=20
#         warmup_steps=0
#         per_device_train_batch_size=128
#         max_source_length=256
#         if [[ "${big_task[@]}" =~ "${task}" ]]; then
#             num_train_epochs=10
#             max_source_length=348
#         fi

#         # if [[ "${small_task[@]}" =~ "${task}" ]]; then
#         #     per_device_train_batch_size=32
#         # fi
#         load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/mnli_fp32/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qnli_fp32/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qqp_fp32/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/sst2_fp32/lora.pt"
#         load_task_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/mnli_fp32/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qnli_fp32/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/qqp_fp32/task_embedding.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage1/sst2_fp32/task_embedding.pt"
#         output_dir="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/result/stage2_softmax_4lora_128/"$learning_rate"_"$task"_"$per_device_train_batch_size"_"$task_reduction_factor

#         echo $task $num_train_epochs
#             python run_seq2seq.py \
#             --do_train=$do_train \
#             --do_eval=$do_eval \
#             --do_test=$do_test \
#             --warmup_steps=$warmup_steps \
#             --save_steps=$save_steps \
#             --save_strategy="$save_strategy" \
#             --model_name_or_path="$model_name_or_path" \
#             --tokenizer_name="$tokenizer_name" \
#             --save_total_limit=$save_total_limit \
#             --per_device_train_batch_size=$per_device_train_batch_size \
#             --per_device_eval_batch_size=$per_device_eval_batch_size \
#             --load_best_model_at_end=$load_best_model_at_end \
#             --metric_for_best_model="$metric_for_best_model" \
#             --greater_is_better=$greater_is_better \
#             --evaluation_strategy="$evaluation_strategy" \
#             --non_linearity="$non_linearity" \
#             --max_source_length=$max_source_length \
#             --learning_rate=$learning_rate \
#             --output_dir="$output_dir" \
#             --split_validation_test=$split_validation_test \
#             --task_name="${t[@]}" \
#             --eval_dataset_name="${t[@]}" \
#             --test_dataset_name="${t[@]}" \
#             --num_train_epochs=$num_train_epochs \
#             --dataset_config_name="${dataset_config_name[@]}" \
#             --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
#             --test_dataset_config_name="${test_dataset_config_name[@]}" \
#             --predict_with_generate=$predict_with_generate \
#             --overwrite_output_dir=$overwrite_output_dir \
#             --compute_memory=$compute_memory \
#             --report_to="$report_to" \
#             --adapter_config_name="adapter" \
#             --train_task_adapters=true \
#             --task_reduction_factor=$task_reduction_factor \
#             --unfreeze_lm_head=false \
#             --unfreeze_layer_norms=true \
#             --add_adapter_in_feed_forward=true \
#             --add_adapter_in_feed_forward_out=false \
#             --add_adapter_in_self_attention=false \
#             --add_layer_norm_before_adapter=false \
#             --add_layer_norm_after_adapter=false \
#             --add_lora=$add_lora \
#             --add_task_embedding=true \
#             --load_task_path=$load_task_path \
#             --logging_steps 10 \
#             --load_lora_path=$load_lora_path
#         bash clean.sh $output_dir
#     done
# done


