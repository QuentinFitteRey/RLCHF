import wandb
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from training.utils import get_sft_dataset_from_persona_preferences


wandb.init(project="rlchf-training-sft")

sft_config = SFTConfig(
    output_dir="../scratch/sft-rlchf-pref",
    logging_steps=25,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_strategy="steps",
    save_steps=1000,
    report_to="wandb",
    bf16=True,
    learning_rate=5e-5,
    max_seq_length=4092,
    dataloader_num_workers=4,
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

datasets = [
    load_dataset(
        "json",
        data_files="../scratch/preference_data/gemma_preferences_gender_align.jsonl",
        split="train",
    ),
    load_dataset(
        "json",
        data_files="../scratch/preference_data/gemma_preferences_social_reasoning.jsonl",
        split="train",
    ),
    load_dataset(
        "json",
        data_files="../scratch/preference_data/gemma_preferences_hh_golden.jsonl",
        split="train",
    ),
]

train_dataset = get_sft_dataset_from_persona_preferences(datasets)


trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
)

trainer.train()

wandb.finish()
