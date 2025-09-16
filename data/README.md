# Data Directory

This directory contains datasets and configuration files for the RLCHF project.

## Structure

```
data/
├── personas/                 # Persona datasets
│   ├── sample_personas.jsonl    # Example persona dataset
│   └── README.md                # This file
├── datasets/                 # Training datasets (create as needed)
├── preferences/             # Generated preference data (create as needed)
└── processed/               # Processed datasets (create as needed)
```

## Usage

### Personas Format

The personas should be in JSONL format with the following structure:

```json
{
  "persona": "A descriptive persona text",
  "sex": "Male/Female/Non-binary",
  "age": 25,
  "nationality": "Nationality",
  "social_class": "Working class/Middle class/Upper class"
}
```

### Getting Started

1. **Download personas**: You can use the provided sample personas or download larger datasets like PersonaHub
2. **Prepare datasets**: Place your training datasets in the `datasets/` folder
3. **Generate preferences**: Use the scripts in `prompt/` to generate preference data
4. **Train models**: Use the scripts in `training/` with your prepared data

### Data Sources

- **PersonaHub**: Large-scale persona dataset for diverse perspectives
- **Anthropic HH-RLHF**: Human preference dataset for alignment training
- **Custom datasets**: Add your own datasets following the expected format
