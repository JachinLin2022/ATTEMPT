#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"
export http_proxy='http://127.0.0.1:7890'
export https_proxy='http://127.0.0.1:7890'
python run_seq2seq.py configs/baselines/adapter.json