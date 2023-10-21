""" T5 model configuration """
from transformers.models.t5 import T5Config


class T5Config(T5Config):
    def __init__(self,
                 train_task_adapters=False,
                 prefix_tuning=False,
                 add_lora=False,
                 lora_num=1,
                 source_task = None,
                 target_task = None,
                 add_task_embedding = None,
                 task_embedding_len = None,
                 task_embedding_init_token = None,
                 load_task_path = None,
                 init_task_from_vocab = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_task_adapters = train_task_adapters
        self.prefix_tuning = prefix_tuning
        self.add_lora = add_lora
        self.lora_num = lora_num
        self.source_task = source_task
        self.target_task = target_task

        # for task embedding
        self.add_task_embedding = add_task_embedding
        self.task_embedding_len = task_embedding_len
        self.task_embedding_init_token = task_embedding_init_token
        self.load_task_path = load_task_path
        self.init_task_from_vocab = init_task_from_vocab