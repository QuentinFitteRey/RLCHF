#!/bin/bash

echo "Generating BBQ dataset..."
python -m eval.benchmarks.bbq.generate_from_template_all_categories
python -m eval.benchmarks.bbq.generate_from_template_intersectional_cats

if [ $? -eq 0 ]; then
    echo "Generation execution completed successfully."
else
    echo "Error occurred during generation."
    exit 1
fi