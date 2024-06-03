import os
import sys
import json
import torch
import logging
import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer
import determined as det
from determined.transformers import DetCallback
from data import download_pach_repo

logger = logging.getLogger(__name__)

# def prepare_compute_metrics(eval_labels_path):
#     def compute_metrics(p):
#         nonlocal eval_labels_path
#         with open(eval_labels_path, "rb") as f:
#             eval_labels = json.load(f)
#     return compute_metrics

def download_data(cluster_info):
    data_config = cluster_info.user_data
    download_directory = (
            f"/tmp/data-rank{cluster_info.container_rank}"
        )
    data_dir = os.path.join(download_directory, "data")

    files = download_pach_repo(
        data_config["pachyderm"]["host"],
        data_config["pachyderm"]["port"],
        data_config["pachyderm"]["repo"],
        data_config["pachyderm"]["branch"],
        data_dir,
        data_config["pachyderm"]["token"],
        data_config["pachyderm"]["project"],
        data_config["pachyderm"]["previous_commit"],
    )
    print(f"Data dir set to : {data_dir}")

    return [des for src, des in files]



def main(training_args, lora_args, det_callback, hparams, training_file):
    # training_data = load_dataset('json', data_files=hparams["train_dataset_path"])['train']
    dataset = load_dataset('json', data_files=training_file)['train']
    dataset = dataset.shuffle()
    train_test_split = dataset.train_test_split(test_size=0.05)
    training_data = train_test_split["train"]
    eval_data = train_test_split["test"]

    # training_data = load_dataset('json', data_files=training_file)['train']
    # if hparams["use_training_data_subset"] is not None:
    #     training_data = training_data.shuffle()
    #     training_data = training_data.select(indices=range(hparams["use_training_data_subset"]))
    # eval_data = load_dataset('json', data_files=hparams["eval_dataset_path"])['train']

    base_model_name = hparams["model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = "(0)"
    tokenizer.padding_side = "right" #Required fro fp16 training

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        use_cache=False)
    
    # Trainer
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        eval_dataset=eval_data,
        peft_config=lora_args,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=hparams["max_seq_length"]
    )

    trainer.add_callback(det_callback)
    trainer.train()
    # trainer.log_metrics("test", metrics={"test_accuracy: 0.8"})


if __name__ == "__main__":
    logging.basicConfig(
        format=det.LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)]
    )
    log_level = logging.INFO
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    info = det.get_cluster_info()
    hparams = info.trial.hparams
    training_file = download_data(info)[0]
    print(f"Training file: {training_file}")

    training_args = TrainingArguments(**hparams["training_args"])
    lora_args = LoraConfig(**hparams["lora_args"])
    if training_args.deepspeed:
        distributed = det.core.DistributedContext.from_deepspeed()
    else:
        distributed = det.core.DistributedContext.from_torch_distributed()
    
    with det.core.init(distributed=distributed) as core_context:
        det_callback = DetCallback(
            core_context,
            training_args
        )
        main(training_args, lora_args, det_callback, hparams, training_file)