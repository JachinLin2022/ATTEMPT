from datasets import load_dataset
from transformers import AutoTokenizer
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# mrpc = load_dataset('squad')
# tokenizer = AutoTokenizer.from_pretrained('t5-base')
data = load_dataset("tau/mrqa", "hotpotqa")
# data = load_dataset("yelp_polarity")