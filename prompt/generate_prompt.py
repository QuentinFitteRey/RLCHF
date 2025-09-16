import numpy as np
from prompt.prompts import (
    system_prompt_cot,
    user_query_cot
)

bot_llama = "<|begin_of_text|>"
start_header_llama = "<|start_header_id|>"
end_header_llama = "<|end_header_llama_id|>"
eoturn_llama = "<|eot_id|>"
eot_llama = "<|end_of_text|>"


def generate_prompt_llama(example, base_prompt, end_prompt, personas, train=False):
    chosen = example["chosen"]
    rejected = example["rejected"]
    context = example["prompt"]

    if np.random.rand() >= 0.5:
        answer_A, answer_B, good = chosen, rejected, "a"
    else:
        answer_A, answer_B, good = rejected, chosen, "b"

    prompts = []
    for persona in personas["persona"]:
        prompt = (
            f"{bot_llama}{start_header_llama}system{end_header_llama}\n"
            f"{base_prompt}\n"
            f"{persona}{eoturn_llama}{start_header_llama}user{end_header_llama}\n"
            f"Query: {context}\n"
            f"Answer A: {answer_A}\n"
            f"Answer B: {answer_B}\n"
            f"{end_prompt}{eoturn_llama}{start_header_llama}assistant{end_header_llama}"
        )
        if train:
            prompt += f"{good}{eot_llama}"
        prompts.append(prompt)
    return prompts, good


bot_gemma = "<bos>"
start_header_gemma = "<start_of_turn>"
eoturn_gemma = "<end_of_turn>"
eot_gemma = "<eos>"


def generate_prompt_gemma(example, base_prompt, end_prompt, personas, train=False):
    chosen = example["chosen"]
    rejected = example["rejected"]
    context = example["prompt"]

    if np.random.rand() >= 0.5:
        answer_A, answer_B, good = chosen, rejected, "a"
    else:
        answer_A, answer_B, good = rejected, chosen, "b"

    prompts = []
    for persona in personas["persona"]:
        prompt = (
            f"{bot_gemma}{start_header_gemma}user\n"
            f"{base_prompt}\n"
            f"{persona}\n"
            f"Query: {context}\n"
            f"Answer A: {answer_A}\n"
            f"Answer B: {answer_B}\n"
            f"{end_prompt}{eoturn_gemma}\n"
            f"{start_header_gemma}model\n"
        )
        if train:
            prompt += f"{good}{eoturn_gemma}{eot_gemma}"
        prompts.append(prompt)
    return prompts, good


bot_qwen = "<|im_start|>"
start_header_qwen = bot_qwen
eoturn_qwen = "<|im_end|>"


def generate_prompt_qwen(example, base_prompt, end_prompt, personas, train=False):
    chosen = example["chosen"]
    rejected = example["rejected"]
    context = example["prompt"]

    if np.random.rand() >= 0.5:
        answer_A, answer_B, good = chosen, rejected, "a"
    else:
        answer_A, answer_B, good = rejected, chosen, "b"

    prompts = []
    for persona in personas["persona"]:
        prompt = (
            f"{start_header_qwen}system\n"
            f"{base_prompt}\n"
            f"{persona}{eoturn_qwen}{start_header_qwen}user\n"
            f"Query: {context}\n"
            f"Answer A: {answer_A}\n"
            f"Answer B: {answer_B}\n"
            f"{end_prompt}{eoturn_qwen}{start_header_qwen}assistant\n"
        )
        if train:
            prompt += f"{good}{eoturn_qwen}"
        prompts.append(prompt)
    return prompts, good


def generate_prompt_cot(example, personas, tokenizer):

    chosen = example["chosen"]
    rejected = example["rejected"]
    query = example["prompt"]

    if np.random.rand() >= 0.5:
        answer_A, answer_B, good = chosen, rejected, "a"
    else:
        answer_A, answer_B, good = rejected, chosen, "b"

    prompts = []
    for persona in personas["persona"]:


        messages = [
            {"role": "system", "content": system_prompt_cot.format(
                persona=persona
                )
            },
            {"role": "user", "content": user_query_cot.format(
                query=query,
                answer_A=answer_A,
                answer_B=answer_B
                )
            }
        ]

        instruct_text = tokenizer.apply_chat_template(messages, tokenize=False)

        prompts.append(instruct_text)

    return prompts, good