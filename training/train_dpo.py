import wandb
import torch
import re
from datasets import load_dataset, concatenate_datasets
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


wandb.init(project="rlchf-training-dpo")


def parse_multiturn(conversation):
    """
    Parses a multi-turn conversation string into a list of message dictionaries.

    Each line is expected to start with "Human:" or "Assistant:".

    Args:
        conversation (str): The multi-turn conversation text.

    Returns:
        list: A list of dictionaries with "role" and "content" keys.
    """
    # Split the conversation into non-empty lines
    lines = [line.strip() for line in conversation.split("\n") if line.strip()]
    messages = []
    for line in lines:
        if line.startswith("Human:"):
            content = line[len("Human:") :].strip()
            messages.append({"role": "user", "content": content})
        elif line.startswith("Assistant:"):
            content = line[len("Assistant:") :].strip()
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": line})
    return messages


def transform_entry(entry):
    """
    Transforms a dataset entry into the required format.

    The 'prompt' field is parsed into multiple turns while 'chosen' and 'rejected'
    are directly wrapped as an assistant's response.

    Args:
        entry (dict): A dictionary with keys "prompt", "chosen", and "rejected".

    Returns:
        dict: A dictionary formatted as:
              {
                  "prompt": [list of message dicts],
                  "chosen": [{"role": "assistant", "content": <chosen_text>}],
                  "rejected": [{"role": "assistant", "content": <rejected_text>}]
              }
    """
    return {
        "prompt": parse_multiturn(entry["prompt"]),
        "chosen": [{"role": "assistant", "content": entry["chosen"]}],
        "rejected": [{"role": "assistant", "content": entry["rejected"]}],
    }


training_args = DPOConfig(
    output_dir="./dpo",
    logging_steps=25,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    metric_for_best_model="eval_loss",
    save_strategy="best",
    report_to="wandb",
    # padding_free=True,
    bf16=True,
    max_prompt_length=2048,
    max_completion_length=2048,
    learning_rate=1e-5, 
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
train_dataset = load_dataset("nz/anthropic-hh-golden-rlhf", split="train")
train_dataset = train_dataset.map(transform_entry)
train_dataset2 = load_dataset("ProlificAI/social-reasoning-rlhf", split="train")
train_dataset2 = train_dataset2.rename_column("question", "prompt")
train_dataset2 = train_dataset2.map(transform_entry)
train_dataset3 = load_dataset(
    "json",
    data_files="data/datasets/GenderAlign_rlhf_format.jsonl",
)["train"]
train_dataset3 = train_dataset3.map(transform_entry)

merged_dataset = concatenate_datasets([train_dataset, train_dataset2, train_dataset3])

merged_dataset = merged_dataset.shuffle(seed=0)


trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,  
    train_dataset=merged_dataset,
)

trainer.train()
