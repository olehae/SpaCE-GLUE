# SpaCE-GLUE

**Spatial Cognition Exercises – General Language Understanding Evaluation**

[![PyPI version](https://badge.fury.io/py/space-glue.svg)](https://pypi.org/project/space-glue/)

A benchmarking framework for evaluating spatial reasoning capabilities of Large Language Models (LLMs).


## Overview

SpaCE-GLUE provides a unified interface for evaluating LLMs on various spatial reasoning benchmarks. It supports:

- Multiple spatial reasoning datasets (bAbI, StepGame, SpartQA, and more)
- OpenAI-compatible APIs and local vLLM inference
- Configurable evaluation workflows with YAML configuration
- Automatic result aggregation and scoring
- Resume capability for interrupted runs


## Installation

### From PyPI

```bash
pip install space-glue
```

### From Source

```bash
git clone https://github.com/your-username/SpaCE-GLUE.git
cd SpaCE-GLUE
pip install -e .
```

### Requirements

- Python >= 3.13
- Dependencies are automatically installed via pip

## Quick Start

1. Create a configuration file `config.yaml`:

```yaml
model:
  class: "models.openai_model.OpenAIModel"
  params:
    name: "gpt-4"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"

datasets:
  - class: "data.SPaRC.SPaRC"

evaluation:
  results_dir: "results"
  batch_size: 1
  runs: [1]
```

2. Set your API key (or use a `.env` file):

```bash
export OPENAI_API_KEY="your-api-key"
```

3. Run the evaluation:

```bash
space-glue --config config.yaml
```


## Configuration

SpaCE-GLUE uses YAML configuration files. Environment variables can be referenced using `${VAR_NAME}` syntax and will be resolved from the environment or a `.env` file.

### Top-Level Structure

```yaml
model:       # Required - Model configuration
datasets:    # Required - List of datasets to evaluate
evaluation:  # Optional - Evaluation settings
logging:     # Optional - Logging configuration
```

---

### `model` (required)

Specifies the model class and its constructor parameters.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class` | string | Yes | Full import path to model class |
| `params` | mapping | No | Constructor arguments for the model |

**Example with OpenAI API:**

```yaml
model:
  class: "models.openai_model.OpenAIModel"
  params:
    name: "gpt-4"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
```

---

### `datasets` (required)

A list of datasets to evaluate. Each entry specifies a dataset class and optional parameters.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class` | string | Yes | Full import path to dataset class |
| `params` | mapping | No | Constructor arguments for the dataset |

**Example:**

```yaml
datasets:
  - class: "data.StepGame.StepGame"
  - class: "data.SpartQA.SpartQA"
  - class: "data.bAbI.bAbI"
```

---

### `evaluation` (optional)

Controls the evaluation workflow behavior.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `results_dir` | string | `"results"` | Directory to store results |
| `batch_size` | int | `1` | Number of prompts to process concurrently |
| `runs` | list[int] | `[1] * len(datasets)` | Number of inference runs per item for each dataset |
| `inference` | bool | `true` | Whether to run model inference and store responnses |
| `evaluate` | bool | `true` | Whether to score responses |
| `aggregate` | bool | `true` | Whether to compute aggregate statistics |

Results are stored as JSONL files in the configured `results_dir`.

**Example:**

```yaml
evaluation:
  results_dir: "my_results"
  batch_size: 5
  runs: [3, 3, 3]      # 3 runs for each of the 3 datasets
  inference: true
  evaluate: true
  aggregate: true
```

---

### `logging` (optional)

Configures logging output.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | string | `"INFO"` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `format` | string | `"%(asctime)s - %(levelname)s - %(message)s"` | Log message format |
| `file` | string | `null` | Optional file path for logging output |

**Example:**

```yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file: "space_glue.log"
```

---

## Available Datasets

SpaCE-GLUE includes the following spatial reasoning benchmarks:

| Dataset | Class Path | Description |
|---------|------------|-------------|
| **bAbI** | `data.bAbI.bAbI` | Facebook's bAbI tasks (spatial subset) |
| **GeoGramBench** | `data.GeoGramBench.GeoGramBench` | Geographic grammar benchmark |
| **GRASP** | `data.GRASP.GRASP` | Grounded Reasoning about Spatial Prepositions |
| **PLUGH** | `data.PLUGH.PLUGH` | Text adventure spatial reasoning |
| **RoomSpace** | `data.RoomSpace.RoomSpace` | Room-based spatial reasoning |
| **SPaRC** | `data.SPaRC.SPaRC` | Spatial Reasoning Compositionally |
| **SpartQA** | `data.SpartQA.SpartQA` | Spatial Question Answering |
| **SpatialEval** | `data.SpatialEval.SpatialEval` | Spatial evaluation benchmark |
| **STBench** | `data.STBench.STBench` | Spatial-Temporal benchmark |
| **StepGame** | `data.StepGame.StepGame` | Multi-hop spatial reasoning |

---

## Available Models

### OpenAIModel

For OpenAI API or compatible endpoints (vLLM, Ollama, etc.).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Model name/identifier |
| `base_url` | string | Yes | API endpoint URL |
| `api_key` | string | Yes | API key (use `"EMPTY"` for local endpoints) |
| `temperature` | float | No | Sampling temperature |
| `reasoning_effort` | string | No | Reasoning effort level (`"low"`, `"medium"`, `"high"`) |

### VLLMModel

For direct local inference using vLLM.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | - | Hugging Face model name |
| `temperature` | float | No | `0.7` | Sampling temperature |

---

## Example Configurations

### Full Benchmark Run

```yaml
# SpaCE-GLUE Evaluation Configuration

# Model Configuration
model:
  class: "models.openai_model.OpenAIModel"
  params:
    name: "gpt-4"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"

# Datasets to evaluate
datasets:
  - class: "data.bAbI.bAbI"
  - class: "data.GeoGramBench.GeoGramBench"
  - class: "data.GRASP.GRASP"
  - class: "data.PLUGH.PLUGH"
  - class: "data.RoomSpace.RoomSpace"
  - class: "data.SPaRC.SPaRC"
  - class: "data.SpartQA.SpartQA"
  - class: "data.SpatialEval.SpatialEval"
  - class : "data.STBench.STBench"
  - class: "data.StepGame.StepGame"

# Evaluation settings
evaluation:
  results_dir: "results"
  batch_size: 5
  runs: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

# Logging settings
logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"

```
