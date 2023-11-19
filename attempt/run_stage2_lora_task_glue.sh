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
add_task_embedding=true
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export http_proxy='http://127.0.0.1:7890'
export https_proxy='http://127.0.0.1:7890'

big_task=(superglue-multirc)
small_task=(superglue-cb superglue-wsc-fixed)
qa_task=(nq newsqa searchqa hotpotqa) 
lrs=(1e-3 3e-4)
task_reduction_factor=16
target_task=(superglue-cb cola superglue-wsc-fixed mrpc stsb rte superglue-wic superglue-boolq superglue-multirc)
target_task=(newsqa searchqa hotpotqa)
target_task=(cola rte superglue-cb superglue-wsc-fixed mrpc stsb superglue-wic superglue-boolq superglue-multirc)

seed=4096
target_task=(yelp_polarity)
echo $CUDA_VISIBLE_DEVICES
for task in ${target_task[@]}
do
    for learning_rate in ${lrs[@]}
    do
        t=($task)
        num_train_epochs=5
        warmup_steps=0
        per_device_train_batch_size=128
        max_source_length=256
        if [[ "${big_task[@]}" =~ "${task}" ]]; then
            num_train_epochs=10
            max_source_length=348
            per_device_train_batch_size=64
        fi

        if [[ "${small_task[@]}" =~ "${task}" ]]; then
            per_device_train_batch_size=32
        fi

        if [[ "${qa_task[@]}" =~ "${task}" ]]; then
            per_device_train_batch_size=32
            per_device_eval_batch_size=128
            max_source_length=512
            num_train_epochs=5
        fi

        if [[ "superglue-boolq" =~ "${task}" ]] || [[ "rte" =~ "${task}" ]]; then
            echo 123123123
            per_device_train_batch_size=100
        fi
        # 4lora
        # load_lora_path="/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/mnli/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qnli/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qqp/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/sst2/lora.pt"
        # load_task_path="/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/mnli/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qnli/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qqp/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/sst2/task_embedding.pt"

        # 6lora init
        load_lora_path="/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/mnli/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qnli/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qqp/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/sst2/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/squad/lora.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/superglue-record/lora.pt"
        load_task_path="/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/mnli/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qnli/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/qqp/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/sst2/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/squad/task_embedding.pt,/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/superglue-record/task_embedding.pt"
        output_dir="/home/linzhisheng/ATTEMPT/attempt/result/stage2_$seed/$task/"$learning_rate"_"$per_device_train_batch_size"_"$task_reduction_factor"_"$CUDA_VISIBLE_DEVICES

        while true
        do
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
                --add_task_embedding=$add_task_embedding \
                --load_task_path=$load_task_path \
                --logging_steps 10 \
                --init_task_from_vocab=true \
                --load_lora_path=$load_lora_path \
                --seed $seed
            if [[ $? -eq 0 ]]; then
                # 如果脚本正常退出，则跳出循环
                break
            fi
        done
        bash clean.sh $output_dir
    done
done