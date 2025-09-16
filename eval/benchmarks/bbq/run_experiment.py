#!/usr/bin/env python3

"""
To run the benchmark evaluation on the BBQ dataset with your custom configuration, run:
    python eval/benchmarks/bbq/run_experiment.py --config_name=myconfig

Make sure you have a configuration file in the `configs/bbq/` directory.
"""

import os
import logging
from tqdm import tqdm
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams

from eval.benchmarks.bbq.utils import format_question, parse_model_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/bbq/",
    config_name="llama3.1-8B-Instruct",
)
def main(cfg: DictConfig):
    """
    Evaluates the BBQ dataset with vLLM using Hydra configs, sending all prompts
    in one generate() call per file without manual batching.
    """
    logger.info("Starting BBQ evaluation with Hydra config.")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pred_dir = cfg.paths.pred_dir
    bbq_path = cfg.paths.bbq_processed

    model_name = cfg.model.name
    temperature = cfg.model.temperature
    max_tokens = cfg.model.max_tokens

    instruction = cfg.inference.instruction
    exp_id = cfg.experiment.id

    logger.info(f"Loading vLLM model: {model_name}")
    llm = LLM(model=model_name, download_dir=cfg.paths.cache_dir)
    logger.info("vLLM model loaded successfully.")

    os.makedirs(os.path.join(pred_dir, exp_id), exist_ok=True)

    cats = [f for f in os.listdir(bbq_path) if f.endswith(".jsonl")]

    for cat in tqdm(cats, desc="Running inference on BBQ categories", unit="category"):
        file_path = os.path.join(bbq_path, cat)
        df = pd.read_json(file_path, lines=True)

        prompts = []
        for _, row in df.iterrows():
            prompt_str = format_question(
                context=row.get("context", ""),
                question=row.get("question", ""),
                ans0=row.get("ans0", ""),
                ans1=row.get("ans1", ""),
                ans2=row.get("ans2", ""),
                instruction=instruction,
            )
            prompts.append(prompt_str)

        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=temperature, max_tokens=max_tokens
            ),
        )

        all_predictions = []
        for output in outputs:
            gen_text = output.outputs[0].text
            predicted_label = parse_model_answer(gen_text)
            all_predictions.append(predicted_label)

        df["prediction"] = all_predictions

        out_path = os.path.join(pred_dir, exp_id, cat)
        df.to_json(out_path, orient="records", lines=True)

    logger.info("All done!")


if __name__ == "__main__":
    main()
