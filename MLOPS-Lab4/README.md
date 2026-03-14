#Large Language Model Data Pipeline Lab

## Overview

This project contains a single Jupyter notebook that demonstrates two practical ways to build a language-modeling data pipeline using the Hugging Face ecosystem.

The notebook uses:
- **Dataset:** `roneneldan/TinyStories`
- **Model:** `distilgpt2`

These choices make the lab easy to run with a fully open, free, and readily available model and dataset.

## Files

- **Normal_and_Streaming_LM_Pipeline.ipynb**  
  A complete lab notebook that shows how to prepare text for causal language modeling in two ways:
  1. a **normal pipeline** using regular dataset loading and preprocessing
  2. a **streaming pipeline** using streamed dataset access for memory-efficient processing

## What the notebook does

The notebook is designed to teach the workflow of preparing a dataset for autoregressive language modeling.

### 1. Environment setup
The notebook installs and imports the required libraries, typically including:
- `datasets`
- `transformers`
- `torch`
- utility libraries for working with tokenization and batching

### 2. Load a different open dataset
Instead of WikiText, this notebook uses **TinyStories**, which is a lightweight text dataset suitable for language-modeling experiments and educational labs.

### 3. Use a free public model
Instead of a larger or more restricted model setup, the notebook uses **DistilGPT-2 (`distilgpt2`)**, which is:
- free to use
- publicly available on Hugging Face
- appropriate for demonstrating causal language modeling pipelines

### 4. Normal preprocessing pipeline
The notebook first demonstrates a standard in-memory preprocessing workflow.

This section generally includes:
- loading a dataset split normally
- inspecting samples
- tokenizing text with the model tokenizer
- grouping or chunking tokens into fixed-length training sequences
- preparing the processed dataset for training or experimentation

This pipeline is useful when:
- the dataset is small or moderate in size
- you want easier debugging and inspection
- you want a simple, beginner-friendly workflow

### 5. Streaming preprocessing pipeline
The notebook then demonstrates a **streaming** version of the pipeline.

This section generally includes:
- loading the dataset in streaming mode
- iterating through samples without downloading the full dataset at once
- tokenizing streamed text examples
- preparing data in a memory-aware way

This pipeline is useful when:
- the dataset is large
- you want to reduce memory usage
- you want to demonstrate scalable dataset handling

### 6. Comparison of both approaches
The notebook is meant to show the difference between:
- a **normal pipeline** that materializes data more directly
- a **streaming pipeline** that processes data incrementally

This helps explain the tradeoff between:
- simplicity and easier inspection
- scalability and memory efficiency

## Why this notebook is useful

This lab is helpful for understanding how modern NLP training pipelines are built using Hugging Face tools.

By working through both normal and streaming versions, you can learn:
- how tokenizers convert text into model inputs
- how causal language-model data is prepared
- how to handle datasets in both standard and scalable ways
- how the same modeling task can use different data-loading strategies

## Suggested use

This notebook is a good fit for:
- coursework labs
- self-learning for Hugging Face datasets and transformers
- introductory experiments in language modeling
- understanding the difference between regular and streaming dataset workflows

## How to run

1. Create and activate a Python environment.
2. Install the required packages:

```bash
pip install datasets transformers torch jupyter
```

3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open:

```text
Normal_and_Streaming_LLM_Pipeline.ipynb
```

5. Run the notebook cells in order.

## Notes

- The notebook focuses on **data pipeline construction** and preprocessing flow.
- Streaming behavior can differ slightly from normal dataset access because items are produced lazily.
- If you want to extend this lab later, you can add:
  - training with `Trainer`
  - evaluation metrics such as perplexity
  - comparison across multiple open models
  - custom chunk lengths and batching strategies
