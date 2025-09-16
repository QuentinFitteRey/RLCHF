from typing import List
from datasets import Dataset, concatenate_datasets, load_dataset


def parse_prompt(prompt):
    messages = []
    parts = prompt.split("Human:")
    for part in parts[1:]:  
        human_part = part.split("Assistant:", 1)[0].strip()
        messages.append({"role": "user", "content": human_part})
        if "Assistant:" in part:
            assistant_part = part.split("Assistant:", 1)[1].strip()
            if assistant_part:
                messages.append({"role": "assistant", "content": assistant_part})
    return messages


def hh_formatting_func(sample):
    prompt = sample["prompt"]
    chosen = sample["chosen"]
    messages = parse_prompt(prompt)
    messages.append({"role": "assistant", "content": chosen})
    return {"messages": messages}


def sr_formatting_func(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["chosen"]},
        ]
    }


def ga_formatting_func(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["chosen"]},
        ]
    }


def preference_formatting_func(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["chosen"]},
        ]
    }


def format_rlhf_dataset_to_sft(dataset, dataset_name):
    if dataset_name == "hh_golden":
        return dataset.map(
            hh_formatting_func,
            remove_columns=dataset.column_names,
        )
    elif dataset_name == "social_reasoning":
        return dataset.map(
            sr_formatting_func,
            remove_columns=dataset.column_names,
        )
    elif dataset_name == "gender_align":
        return dataset.map(
            ga_formatting_func,
            remove_columns=dataset.column_names,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")




def gemma_to_messages(gemma_text: str):

    text = gemma_text.replace("<bos>", "").strip()
    segments = text.split("<start_of_turn>")

    conversation = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        parts = seg.split("\n", 1)
        role = parts[0].strip()

        content = parts[1] if len(parts) > 1 else ""
        content = content.split("<end_of_turn>")[0].strip()

        if role == "user":
            role = "user"
        elif role == "model":
            role = "assistant"

        conversation.append({"role": role, "content": content})

    conversation = [entry for entry in conversation if entry["content"]]

    return conversation


def build_conversation(example):
    prompt = example["prompt_gen"]
    response = example["raw_output"]
    messages = gemma_to_messages(prompt)
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages}


def get_sft_dataset_from_persona_preferences(datasets: List[Dataset]):

    merged_dataset = concatenate_datasets(datasets)

    shuffled_dataset = merged_dataset.shuffle(seed=0)

    sft_dataset = shuffled_dataset.map(
        build_conversation,
        remove_columns=shuffled_dataset.column_names,
    )

    sft_dataset = sft_dataset.filter(lambda x: len(x["messages"]) > 0)
    sft_dataset = sft_dataset.filter(
        lambda x: all(msg["content"] for msg in x["messages"])
    )
    sft_dataset = sft_dataset.filter(
        lambda x: all(msg["role"] for msg in x["messages"])
    )

    return sft_dataset


def format_persona_dpo_dataset(sample):
    model_answer = sample["answer"].lower()   
    gt_answer    = sample["gt_answer"].lower()
    is_correct   = model_answer == gt_answer

    persona = sample["persona"]
    prompt  = sample["prompt"]

    if model_answer == "c" or is_correct:
        good = sample["chosen"]     
        bad  = sample["rejected"]
    else:
        good = sample["rejected"]
        bad  = sample["chosen"]

    if model_answer == "a":
        good_arg = sample["args_for_a"]
        bad_arg  = sample["args_for_b"]

    elif model_answer == "b":
        good_arg = sample["args_for_b"]
        bad_arg  = sample["args_for_a"]

    elif model_answer == "c":
        if gt_answer == "a":
            good_arg = sample["args_for_a"]
            bad_arg  = sample["args_for_b"]
        else:                      
            good_arg = sample["args_for_b"]
            bad_arg  = sample["args_for_a"]
    else:                         
        good_arg = ""
        bad_arg  = ""

    return {
        "prompt": [
            {"role": "system", "content": "You are: " + persona},
            {"role": "user",   "content": prompt},
        ],
        "chosen": [
            {
                "role": "assistant",
                "content": good + "\nArgument for this answer: " + good_arg
            },
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": bad + "\nArgument for this answer: " + bad_arg
            },
        ],
    }



def format_persona_dpo_reason_dataset(sample):

    model_answer = sample["answer"].lower()
    gt_anwer = sample["gt_answer"].lower()
    is_correct = model_answer == gt_anwer

    persona = sample["persona"]
    prompt = sample["prompt"]

    if model_answer == "c" or is_correct:
        good = sample["chosen"]
        bad = sample["rejected"]
    else:
        good = sample["rejected"]
        bad = sample["chosen"]

    return {
        "prompt": [
            {"role": "system", "content": "You are: " + persona},
            {"role": "user", "content": prompt},
        ],
        "chosen": [
            {"role": "assistant", "content": good},
        ],
        "rejected": [
            {"role": "assistant", "content": bad},
        ],
    }
