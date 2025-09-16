import wandb
import torch
import re
from datasets import load_dataset, concatenate_datasets, load_from_disk
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from training.utils import format_persona_dpo_dataset


wandb.init(project="rlchf-training-persona-dpo")


training_args = DPOConfig(
    output_dir="outputs/dpo_llama_persona_cot",
    logging_steps=25,
    num_train_epochs=1,
    per_device_train_batch_size=6,
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    save_steps=1000,
    report_to="wandb",
    bf16=True,
    max_prompt_length=2048,
    max_completion_length=2048,
    gradient_accumulation_steps=4,
    dataloader_num_workers=6,
    dataset_num_proc=6,
    learning_rate=1e-5,  
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
     #"Qwen/Qwen2.5-7B-Instruct",
    #"google/gemma-3-12b-it",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    force_download=True,
)
model.config.use_cache = False

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
tokenizer.padding_side  = 'left'


datasets = [
    load_dataset(
        "json",
        data_files="data/dataset/preference/gemma_preferences_hh_gender_cleaned.jsonl",
        split="train",
    ),
    load_dataset(
        "json",
        data_files="data/dataset/preference/gemma_preferences_hh_reasoning_cleaned.jsonl",
        split="train",
    ),
    load_dataset(
        "json",
        data_files="data/dataset/preference/gemma_preferences_hh_golden_cleaned.jsonl",
        split="train",
    ),
]

merged_dataset = concatenate_datasets(datasets)

merged_dataset = merged_dataset.shuffle(seed=0)
merged_dataset = merged_dataset.filter(
    lambda example: len(example["chosen"]) > 0 
    and len(example["rejected"]) > 0 
    and len(example["prompt"]) > 0 
    and len(example["persona"]) > 0
)

formatted_dataset = merged_dataset.map(
    format_persona_dpo_dataset,
    remove_columns=merged_dataset.column_names,
)


trainer = DPOTrainer(
    model=model,  
    args=training_args,
    processing_class=tokenizer,  
    train_dataset=merged_dataset,
)

# Train the model
# trainer.train(resume_from_checkpoint="data/persona-dpo-qwen/checkpoint-6000")
trainer.train()
