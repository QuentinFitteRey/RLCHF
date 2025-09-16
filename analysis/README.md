# Analysis

This directory contains analysis notebooks and visualizations for the RLCHF project.

## Contents

### Notebooks
- `preference_analysis.ipynb` - Analysis of preference data and persona diversity
- `visualization.ipynb` - Visualization tools and plots for results

### Visualizations
- `accuracy_per_persona.png` - Accuracy breakdown by different personas
- `disagreement_with_majority.png/.pdf` - Analysis of disagreement patterns
- `winogender_evaluation_comparison.png` - Gender bias evaluation comparison

## Usage

1. **Install Jupyter dependencies** (if not already installed):
   ```bash
   pip install jupyter matplotlib seaborn
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook analysis/
   ```

3. **Run the notebooks** to reproduce the analysis and visualizations

## Data Requirements

The notebooks expect data in the following structure:
- Preference data in `data/preferences/` 
- Model outputs in `data/outputs/`
- Evaluation results in appropriate subdirectories

Make sure to update paths in the notebooks if your data is stored elsewhere.

## Analysis Overview

The analysis covers:
- **Persona Diversity**: How different personas affect model responses
- **Preference Agreement**: Consensus and disagreement patterns
- **Bias Evaluation**: Gender and social bias measurements
- **Performance Metrics**: Accuracy and alignment metrics
