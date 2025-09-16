import random
import os
import json
import argparse
import pandas as pd
from vllm import LLM
from dotenv import load_dotenv
from vllm.sampling_params import SamplingParams, GuidedDecodingParams

# Constants
SEX = ["female", "male"]
AGE = ["child", "teenager", "young adult", "adult", "elderly"]
NATIONALITIES = [
    "Brazilian",
    "Nigerian",
    "Japanese",
    "Egyptian",
    "Canadian",
    "Mexican",
    "Swedish",
    "Indian",
    "Turkish",
    "Kenyan",
    "Vietnamese",
    "French",
    "Filipino",
    "Peruvian",
    "Ethiopian",
    "South Korean",
    "German",
    "Australian",
    "South African",
    "Jamaican",
    "Russian",
    "Indonesian",
    "Argentinian",
    "Moroccan",
    "Italian",
    "Bangladeshi",
    "Lebanese",
    "Chinese",
    "Chilean",
    "Spanish",
    "American",
]
SOCIAL_CLASSES = ["lower", "middle", "upper"]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "sex": {"type": "string"},
        "age": {"type": "string"},
        "nationality": {"type": "string"},
        "social_class": {"type": "string"},
    },
    "required": ["description", "sex", "age", "nationality", "social_class"],
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate personas using an LLM.")

    parser.add_argument(
        "--input-file",
        type=str,
        default="data/personas/persona.jsonl",
        help="Path to the input JSONL file containing seed personas",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="data/personas/generated_personas.jsonl",
        help="Path to save the generated personas",
    )

    parser.add_argument(
        "--sample-size", type=int, default=1000, help="Number of personas to generate"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name of the LLM model to use",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for the LLM",
    )

    parser.add_argument(
        "--random-seed", type=int, default=1, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM sampling (higher = more creative)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens in generated responses",
    )

    return parser.parse_args()


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv(".env")
    hf_token = os.environ.get("HF_TOKEN")
    hf_home = os.environ.get("HF_HOME")

    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
    if not hf_home:
        print("Warning: HF_HOME not found in environment variables")

    return hf_token, hf_home


def load_seed_personas(filepath, sample_size=100, random_state=1):
    """Load and sample persona data from a jsonl file."""
    try:
        df = pd.read_json(filepath, lines=True)
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        df = df.rename(columns={"persona": "seed_persona"})
        return df
    except Exception as e:
        print(f"Error loading seed personas: {e}")
        raise


def create_persona_prompt(persona_desc, sex, age, nationality, social_class):
    """Create a prompt for persona generation."""
    prompt = (
        f"Generate a detailed persona description in JSON format based on the following information.\n\n"
        f"Persona Description: {persona_desc}\n"
        f"Sex: {sex}\n"
        f"Age: {age}\n"
        f"Nationality: {nationality}\n"
        f"Social Class: {social_class}\n\n"
        "The persona description should be a few sentences long and include as much details as possible.\n"
        "on the persona's background, interests, hobbies, goals, and challenges. It should specifically develop\n"
        "the persona's social characteristics, such as personality traits, values, attitudes, and behaviors.\n\n"
        "Please make sure that the persona description is coherent, if some information are conflicting, please\n"
        "choose the most appropriate one.\n\n"
        "You will answer this question in JSON format, with the following keys:\n"
        "- description: a few sentences long description of the persona\n"
        "- sex: sex of the persona\n"
        "- age: age group of the persona\n"
        "- nationality: where the persona is from\n"
        "- social_class: social class of the persona\n\n"
        "ONLY OUTPUT THE JSON OBJECT, DO NOT INCLUDE ANYTHING ELSE IN YOUR ANSWER."
    )
    return prompt


def prepare_prompts(df, random_seed=1):
    """Generate prompts for each row in the dataframe with random attributes."""
    # Set random seed for reproducibility
    random.seed(random_seed)

    persona_descriptions = df["seed_persona"].tolist()

    for index, _ in df.iterrows():
        sex = random.choice(SEX)
        age = random.choice(AGE)
        nationality = random.choice(NATIONALITIES)
        social_class = random.choice(SOCIAL_CLASSES)
        persona_desc = random.choice(persona_descriptions)

        prompt = create_persona_prompt(
            persona_desc=persona_desc,
            sex=sex,
            age=age,
            nationality=nationality,
            social_class=social_class,
        )

        df.at[index, "prompt"] = prompt

    return df


def initialize_llm(
    model_name="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=2
):
    """Initialize the LLM with specified parameters."""
    try:
        return LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise


def generate_personas(llm, prompts, json_schema, temperature=0.7, max_tokens=1000):
    """Generate personas using the LLM with guided decoding."""
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        guided_decoding=guided_decoding_params,
    )

    try:
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]
    except Exception as e:
        print(f"Error generating personas: {e}")
        raise


def parse_json_outputs(json_strings):
    """Parse JSON strings into Python objects, handling errors."""
    parsed_data = []
    for i, json_str in enumerate(json_strings):
        try:
            json_obj = json.loads(json_str.strip())
            parsed_data.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON at index {i}: {e}")
            print(f"Problematic string: {json_str}")

    return parsed_data


def save_to_jsonl(data, output_file):
    """Save parsed data to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Successfully saved {len(data)} personas to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise


def main():
    """Main function to orchestrate the persona generation process."""
    args = parse_arguments()

    load_environment()

    df = load_seed_personas(args.input_file, args.sample_size, args.random_seed)

    df = prepare_prompts(df, args.random_seed)

    llm = initialize_llm(args.model_name, args.tensor_parallel_size)

    json_outputs = generate_personas(
        llm, df["prompt"].tolist(), JSON_SCHEMA, args.temperature, args.max_tokens
    )

    parsed_data = parse_json_outputs(json_outputs)

    save_to_jsonl(parsed_data, args.output_file)

    return parsed_data


if __name__ == "__main__":
    main()
