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
# target_task=(superglue-cb superglue-wsc-fixed cola rte)
# target_task=(cola rte mrpc stsb)
# target_task=(superglue-wic)
big_task=(superglue-multirc)
small_task=(superglue-cb superglue-wsc-fixed)
# loras=(mnli qnli sst2 qqp superglue-record)
lrs=(3e-4 1e-3 6e-4)


# 单个lora

target_task=(superglue-multirc superglue-cb superglue-wsc-fixed superglue-boolq mrpc stsb)
# loras=(mnli qnli)
# for lora in ${loras[@]}
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
#             max_source_length=384
#         fi
#         if [[ "${small_task[@]}" =~ "${task}" ]]; then
#             per_device_train_batch_size=32
#         fi
#         load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora2/lora_"$lora"/lora.pt"
#         output_dir="outputs/out_moe/load_"$lora"_"$task
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
#             --task_reduction_factor=16 \
#             --unfreeze_lm_head=false \
#             --unfreeze_layer_norms=true \
#             --add_adapter_in_feed_forward=true \
#             --add_adapter_in_feed_forward_out=false \
#             --add_adapter_in_self_attention=false \
#             --add_layer_norm_before_adapter=false \
#             --add_layer_norm_after_adapter=false \
#             --add_lora=$add_lora \
#             --logging_steps 10 \
#             --load_lora_path=$load_lora_path
#     done
# done

target_task=(superglue-wic)
# lora moe，不同lr
for learning_rate in ${lrs[@]}
do
    bash clean.sh
    for task in ${target_task[@]}
    do
        t=($task)
        num_train_epochs=20
        warmup_steps=0
        per_device_train_batch_size=32
        max_source_length=256
        if [[ "${big_task[@]}" =~ "${task}" ]]; then
            num_train_epochs=10
            max_source_length=384
        fi

        if [[ "${small_task[@]}" =~ "${task}" ]]; then
            per_device_train_batch_size=32
        fi
        load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora2/lora_mnli/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora2/lora_qnli/lora.pt,/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora2/lora_qqp/lora.pt"
        output_dir="outputs/out_mnli_qnli_qqp/load_"$learning_rate"_"$task"_"$per_device_train_batch_size

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
            --task_reduction_factor=16 \
            --unfreeze_lm_head=false \
            --unfreeze_layer_norms=true \
            --add_adapter_in_feed_forward=true \
            --add_adapter_in_feed_forward_out=false \
            --add_adapter_in_self_attention=false \
            --add_layer_norm_before_adapter=false \
            --add_layer_norm_after_adapter=false \
            --add_lora=$add_lora \
            --logging_steps 10 \
            --load_lora_path=$load_lora_path
    done
done



# for task in ${target_task[@]}
# do
#     bash clean.sh
#     for lora in ${loras[@]}
#     do
#         t=($task)
#         num_train_epochs=20
#         warmup_steps=0
#         per_device_train_batch_size=128
#         if [[ "${big_task[@]}" =~ "${task}" ]]; then
#             num_train_epochs=10
#         fi

#         if [[ "${small_task[@]}" =~ "${task}" ]]; then
#             per_device_train_batch_size=32
#         fi
#         output_dir="outputs/out_lora_no_prefix/load_"$lora"_"$task
#         load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora2/lora_"$lora"/lora.pt"

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
#             --task_reduction_factor=16 \
#             --unfreeze_lm_head=false \
#             --unfreeze_layer_norms=true \
#             --add_adapter_in_feed_forward=true \
#             --add_adapter_in_feed_forward_out=false \
#             --add_adapter_in_self_attention=false \
#             --add_layer_norm_before_adapter=false \
#             --add_layer_norm_after_adapter=false \
#             --add_lora=$add_lora \
#             --logging_steps 10 \
#             --load_lora_path=$load_lora_path
#     done
# done


# loras=(qnli)
# for task in ${target_task[@]}
# do
#     bash clean.sh
#     for lora in ${loras[@]}
#     do
#         lora=mix
#         t=($task)
#         num_train_epochs=20
#         warmup_steps=0
#         if [[ "${big_task[@]}" =~ "${task}" ]]; then
#             num_train_epochs=10
#         fi
#         output_dir="outputs/out_lora_no_prefix/load_"$lora"_"$task
#         load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/mix/lora.pt"

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
#             --task_reduction_factor=16 \
#             --unfreeze_lm_head=false \
#             --unfreeze_layer_norms=true \
#             --add_adapter_in_feed_forward=true \
#             --add_adapter_in_feed_forward_out=false \
#             --add_adapter_in_self_attention=false \
#             --add_layer_norm_before_adapter=false \
#             --add_layer_norm_after_adapter=false \
#             --add_lora=$add_lora \
#             --logging_steps 10 \
#             --load_lora_path=$load_lora_path
#     done
# done

# for task in ${target_task[@]}
# do
#     bash clean.sh
#     for lora in ${loras[@]}
#     do
#         t=($task)
#         num_train_epochs=20
#         warmup_steps=0
#         if [[ "${big_task[@]}" =~ "${task}" ]]; then
#             num_train_epochs=10
#         fi
#         output_dir="outputs/out_lora/load_no_"$lora"_"$task
#         load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora/lora_"$lora"/lora.pt"

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
#             --task_reduction_factor=16 \
#             --unfreeze_lm_head=false \
#             --unfreeze_layer_norms=true \
#             --add_adapter_in_self_attention=false \
#             --add_layer_norm_before_adapter=false \
#             --add_layer_norm_after_adapter=false \
#             --add_lora=false \
#             --logging_steps 10 \
#             --load_lora_path=$load_lora_path
#     done
# done


# target_task=(superglue-cb superglue-wic superglue-wsc-fixed superglue-multirc superglue-boolq)
# big_task=(superglue-multirc superglue-boolq)
# loras=(superglue-record qnli qqp sst2)
# for lora in ${loras[@]}
# do
#     bash clean.sh
#     for task in ${target_task[@]}
#     do
        
#         t=($task)
#         num_train_epochs=20
#         warmup_steps=0
#         per_device_train_batch_size=32
#         if [[ "${big_task[@]}" =~ "${task}" ]]; then
#             num_train_epochs=10
#             per_device_train_batch_size=128
#         fi

#         output_dir="outputs/padapter_5e-4/load_"$lora"_"$task
#         load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora_"$lora"/lora.pt"

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
#             --task_reduction_factor=16 \
#             --unfreeze_lm_head=false \
#             --unfreeze_layer_norms=false \
#             --add_adapter_in_self_attention=false \
#             --add_layer_norm_before_adapter=false \
#             --add_layer_norm_after_adapter=false \
#             --add_lora=$add_lora \
#             --logging_steps 10 \
#             --load_lora_path=$load_lora_path
#     done
# done

# for task in ${target_task[@]}
# do
    
#     t=($task)
#     num_train_epochs=20
#     if [[ "${big_task[@]}" =~ "${task}" ]]; then
#         num_train_epochs=10
#     fi
#     output_dir="outputs/padapter_norecordlora_"$task
#     echo $task $num_train_epochs
#         python run_seq2seq.py \
#         --do_train=$do_train \
#         --do_eval=$do_eval \
#         --do_test=$do_test \
#         --warmup_steps=$warmup_steps \
#         --save_steps=$save_steps \
#         --save_strategy="$save_strategy" \
#         --model_name_or_path="$model_name_or_path" \
#         --tokenizer_name="$tokenizer_name" \
#         --save_total_limit=$save_total_limit \
#         --per_device_train_batch_size=$per_device_train_batch_size \
#         --per_device_eval_batch_size=$per_device_eval_batch_size \
#         --load_best_model_at_end=$load_best_model_at_end \
#         --metric_for_best_model="$metric_for_best_model" \
#         --greater_is_better=$greater_is_better \
#         --evaluation_strategy="$evaluation_strategy" \
#         --non_linearity="$non_linearity" \
#         --max_source_length=$max_source_length \
#         --learning_rate=$learning_rate \
#         --output_dir="$output_dir" \
#         --split_validation_test=$split_validation_test \
#         --task_name="${t[@]}" \
#         --eval_dataset_name="${t[@]}" \
#         --test_dataset_name="${t[@]}" \
#         --num_train_epochs=$num_train_epochs \
#         --dataset_config_name="${dataset_config_name[@]}" \
#         --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
#         --test_dataset_config_name="${test_dataset_config_name[@]}" \
#         --predict_with_generate=$predict_with_generate \
#         --overwrite_output_dir=$overwrite_output_dir \
#         --compute_memory=$compute_memory \
#         --report_to="$report_to" \
#         --adapter_config_name="adapter" \
#         --train_task_adapters=true \
#         --task_reduction_factor=16 \
#         --unfreeze_lm_head=false \
#         --unfreeze_layer_norms=false \
#         --add_adapter_in_self_attention=false \
#         --add_layer_norm_before_adapter=false \
#         --add_layer_norm_after_adapter=false \
#         --add_lora=false \
#         --load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora_superglue-record/lora.pt"
# done



# for task in ${target_task[@]}
# do
    
#     t=($task)
#     num_train_epochs=20
#     if [[ "${big_task[@]}" =~ "${task}" ]]; then
#         num_train_epochs=10
#     fi
#     output_dir="outputs/padapter_loadmnli_"$task
#     echo $task $num_train_epochs
#         python run_seq2seq.py \
#         --do_train=$do_train \
#         --do_eval=$do_eval \
#         --do_test=$do_test \
#         --warmup_steps=$warmup_steps \
#         --save_steps=$save_steps \
#         --save_strategy="$save_strategy" \
#         --model_name_or_path="$model_name_or_path" \
#         --tokenizer_name="$tokenizer_name" \
#         --save_total_limit=$save_total_limit \
#         --per_device_train_batch_size=$per_device_train_batch_size \
#         --per_device_eval_batch_size=$per_device_eval_batch_size \
#         --load_best_model_at_end=$load_best_model_at_end \
#         --metric_for_best_model="$metric_for_best_model" \
#         --greater_is_better=$greater_is_better \
#         --evaluation_strategy="$evaluation_strategy" \
#         --non_linearity="$non_linearity" \
#         --max_source_length=$max_source_length \
#         --learning_rate=$learning_rate \
#         --output_dir="$output_dir" \
#         --split_validation_test=$split_validation_test \
#         --task_name="${t[@]}" \
#         --eval_dataset_name="${t[@]}" \
#         --test_dataset_name="${t[@]}" \
#         --num_train_epochs=$num_train_epochs \
#         --dataset_config_name="${dataset_config_name[@]}" \
#         --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
#         --test_dataset_config_name="${test_dataset_config_name[@]}" \
#         --predict_with_generate=$predict_with_generate \
#         --overwrite_output_dir=$overwrite_output_dir \
#         --compute_memory=$compute_memory \
#         --report_to="$report_to" \
#         --adapter_config_name="adapter" \
#         --train_task_adapters=true \
#         --task_reduction_factor=16 \
#         --unfreeze_lm_head=false \
#         --unfreeze_layer_norms=false \
#         --add_adapter_in_self_attention=false \
#         --add_layer_norm_before_adapter=false \
#         --add_layer_norm_after_adapter=false \
#         --add_lora=$add_lora \
#         --load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora_mnli/lora.pt"
# done

# for task in ${target_task[@]}
# do
    
#     t=($task)
#     num_train_epochs=20
#     if [[ "${big_task[@]}" =~ "${task}" ]]; then
#         num_train_epochs=10
#     fi
#     output_dir="outputs/padapter_nomnlilora_"$task
#     echo $task $num_train_epochs
#         python run_seq2seq.py \
#         --do_train=$do_train \
#         --do_eval=$do_eval \
#         --do_test=$do_test \
#         --warmup_steps=$warmup_steps \
#         --save_steps=$save_steps \
#         --save_strategy="$save_strategy" \
#         --model_name_or_path="$model_name_or_path" \
#         --tokenizer_name="$tokenizer_name" \
#         --save_total_limit=$save_total_limit \
#         --per_device_train_batch_size=$per_device_train_batch_size \
#         --per_device_eval_batch_size=$per_device_eval_batch_size \
#         --load_best_model_at_end=$load_best_model_at_end \
#         --metric_for_best_model="$metric_for_best_model" \
#         --greater_is_better=$greater_is_better \
#         --evaluation_strategy="$evaluation_strategy" \
#         --non_linearity="$non_linearity" \
#         --max_source_length=$max_source_length \
#         --learning_rate=$learning_rate \
#         --output_dir="$output_dir" \
#         --split_validation_test=$split_validation_test \
#         --task_name="${t[@]}" \
#         --eval_dataset_name="${t[@]}" \
#         --test_dataset_name="${t[@]}" \
#         --num_train_epochs=$num_train_epochs \
#         --dataset_config_name="${dataset_config_name[@]}" \
#         --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
#         --test_dataset_config_name="${test_dataset_config_name[@]}" \
#         --predict_with_generate=$predict_with_generate \
#         --overwrite_output_dir=$overwrite_output_dir \
#         --compute_memory=$compute_memory \
#         --report_to="$report_to" \
#         --adapter_config_name="adapter" \
#         --train_task_adapters=true \
#         --task_reduction_factor=16 \
#         --unfreeze_lm_head=false \
#         --unfreeze_layer_norms=false \
#         --add_adapter_in_self_attention=false \
#         --add_layer_norm_before_adapter=false \
#         --add_layer_norm_after_adapter=false \
#         --add_lora=false \
#         --load_lora_path="/mlx_devbox/users/linzhisheng.2021/ATTEMPT/attempt/outputs/lora_mnli/lora.pt"
# done