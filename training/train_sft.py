import wandb
import torch
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from training.utils import format_rlhf_dataset_to_sft


wandb.init(project="rlchf-training-sft")

sft_config = SFTConfig(
    output_dir="../scratch/sft-rlchf",
    logging_steps=25,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_strategy="steps",
    save_steps=1000,
    report_to="wandb",
    bf16=True,
    learning_rate=5e-5,
    max_seq_length=4092,
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

hh_golden = load_dataset("nz/anthropic-hh-golden-rlhf", split="train")
hh_golden = format_rlhf_dataset_to_sft(dataset=hh_golden, dataset_name="hh_golden")

social_reasoning = load_dataset("ProlificAI/social-reasoning-rlhf", split="train")
social_reasoning = format_rlhf_dataset_to_sft(
    dataset=social_reasoning, dataset_name="social_reasoning"
)

gender_align = load_dataset(
    "json",
    data_files="data/rlhf_datasets/GenderAlign_rlhf_format.jsonl",
)["train"]
gender_align = format_rlhf_dataset_to_sft(
    dataset=gender_align,
    dataset_name="gender_align",
)

train_dataset = concatenate_datasets([hh_golden, social_reasoning, gender_align])


train_dataset = train_dataset.shuffle(seed=0)


trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
)

trainer.train()

wandb.finish()
