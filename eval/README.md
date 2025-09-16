# Evaluation

## BBQ

> [!IMPORTANT]
> Before you begin, set your `PYTHONPATH` to the project’s root directory:
```bash
export PYTHONPATH=$PWD
```

### Preparing the Data

First, download the BBQ dataset by running the following script:

> [!NOTE]
> This script uses the `DATA_DIR` variable from your `.env` file. If needed, update `.env` to change where the dataset is stored.

```bash
python scripts/download_bbq_dataset.py
```


### Run the Benchmark on Your Model
After you have generated all the benchmark data, run the evaluation on your model:
```bash
python eval/benchmarks/bbq/run_experiment.py --config-name=llama3.1-8B-Instruct
```
> [!NOTE]
> The inference is done with VLLM for faster computation. Batch size is automatically adjusted according to your GPU memory and the model size you are evaluating.

### Computing Bias Scores

When inference completes, compute the bias scores using the adapted script from the [BBQ github repository](https://github.com/nyu-mll/BBQ/blob/main/analysis_scripts/BBQ_calculate_bias_score.R):

```bash
python eval/benchmarks/bbq/bias_score_analysis.py \
--data_dir data/predictions/bbq/meta-llama/Llama-3.1-8B-Instruct/
```
This command automatically saves heatmap plots to the `data_dir` you specified.
