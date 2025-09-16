import argparse
import torch
import re
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
from prompt.generate_prompt import (
    generate_prompt_cot
)
from prompt.prompts import JSON_SCHEMA


def load_data(data_path, personas_path, num_personas):
    print(data_path)
    if "social-reasoning-rlhf" in data_path or "GenderAlign" in data_path:
        df = pd.read_json(data_path, lines=True)
        df = df.rename(
            columns={"chosen": "chosen", "question": "prompt", "rejected": "rejected"}
        )
    else:
        df = pd.read_parquet(data_path)
    personas = pd.read_json(personas_path, lines=True).iloc[:num_personas]
    if "description" in personas.columns:
        personas = personas.rename(columns={"description": "persona"})
    return df, personas


def parse_model_answer(text):
    text = text.lower()
    match = re.search(r"answer:\s([abc])", text)
    if match:
        return match.group(1)
    return None


def hotpot_qa_parser(response: str):
    answer = re.findall(r"<Ans>(.*?)</Ans>", response)
    if len(answer) == 0:
        return None
    return answer[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate persona-based prompts and run inference with vLLM."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset file (parquet format).",
    )
    parser.add_argument(
        "--personas_path",
        type=str,
        required=True,
        help="Path to the personas JSONL file.",
    )
    parser.add_argument(
        "--num_personas",
        type=int,
        default=5,
        help="Number of personas to use from the personas file.",
    )
    parser.add_argument(
        "--first_sample_idx",
        type=int,
        default=0,
        help="Index of the first sample to generate.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or path for vLLM."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the json containing the predictions",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Maximum tokens to generate."
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum concatenated model input length. (for vLLM)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism. (for vLLM)",
    )

    args = parser.parse_args()

    df, personas = load_data(args.data_path, args.personas_path, args.num_personas)

    df = df.iloc[
        args.first_sample_idx : min(args.first_sample_idx + args.num_samples, len(df))
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    df[["prompt_gen", "gt_answer"]] = df.apply(
        lambda x: generate_prompt_cot(
            x, personas, tokenizer,
        ),
        axis=1,
        result_type="expand",
    )

    df["persona"] = df["prompt_gen"].apply(lambda x: personas["persona"].tolist())
    df["sex"] = df["prompt_gen"].apply(lambda x: personas["sex"].tolist())
    df["age"] = df["prompt_gen"].apply(lambda x: personas["age"].tolist())
    df["nationality"] = df["prompt_gen"].apply(
        lambda x: personas["nationality"].tolist()
    )
    df["social_class"] = df["prompt_gen"].apply(
        lambda x: personas["social_class"].tolist()
    )

    df = df.explode(
        ["prompt_gen", "persona", "sex", "age", "nationality", "social_class"],
        ignore_index=True,
    )

    llm = LLM(
        model=args.model,
        dtype=torch.bfloat16,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens,
        guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA),
    )

    outputs = llm.generate(
        df["prompt_gen"].to_list(), sampling_params=sampling_params, use_tqdm=True
    )

    df["raw_output"] = [o.outputs[0].text for o in outputs]

    df.to_json(
        path_or_buf=args.output_path.replace(".json", "_raw.json"),
        orient="records",
        lines=True,
    )
    df["parsed"] = df["raw_output"].apply(json.loads)   

    wanted_keys = [
        "args_for_a",
        "args_for_b",
        "age",
        "preference_reasoning",
        "answer"
    ]

    df[wanted_keys] = df["parsed"].apply(
        lambda d: pd.Series({k: d.get(k) for k in wanted_keys})
    )

    df["answer"] = df["answer"].apply(hotpot_qa_parser)

    df.drop(columns=["parsed"], inplace=True)

    df.to_json(path_or_buf=args.output_path, orient="records", lines=True)



if __name__ == "__main__":
    main()
