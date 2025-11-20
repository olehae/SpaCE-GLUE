# SpaCE-GLUE


## Configuration (`config.yaml`)

This project reads runtime settings from a YAML configuration file. The following
documents the recommended format, the expected data types, which fields are
required, and default values used by the code when a field is omitted.

- **Top-level keys (summary)**
  - `model` (required): Mapping describing the model class and constructor parameters.
  - `datasets` (required, non-empty): Sequence of dataset entries to evaluate.
  - `evaluation` (optional): Settings for evaluation / results handling.
  - `logging` (optional): Logging configuration.

- Any string value set precisely in the form `${VAR_NAME}` will be resolved
from the environment when the config is loaded. The loader calls `load_dotenv()`
so values from a `.env` file are also considered. If the environment variable
is not set, config loading will raise `ValueError`.
- The model´s and each dataset entry's `class` is expected to be a full import path string (e.g. `package.module.ClassName`). The runner will call the configured loader to import and instantiate classes using the `params` mapping.

### `model` (required)

- Type: mapping/dict
- Structure (recommended):

```yaml
model:
  class: "models.openai_model.OpenAIModel"  # string, required: full import path to model class
  params:                                   # mapping of constructor args (optional)
    name: "gpt-4"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
```


### `datasets` (required, non-empty)

- Type: sequence (list) of mappings
- Each entry (recommended format):

```yaml
datasets:
  - class: "data.example_dataset.ExampleDataset"  # string, required: full import path to dataset class
    params:                                       # mapping of constructor args (optional)
      limit: 20                                    
```


### `evaluation` (optional)

- Type: mapping/dict
- Defaults:
  - `results_dir`: string, default: "results"
  - `save_every`: int, default: 50
  - `batch_size`: int, default: 1
  - `inference`: Bool flag; default `true`.
  - `scoring`: Bool flag; default `true`.

Example:

```yaml
evaluation:
  results_dir: "my_results"
  save_every: 10
  batch_size: 1
  inference: false
  scoring: true
```

### `logging` (optional)

- Type: mapping/dict
- Defaults:
  - `level`: string, default: "INFO" (e.g. "DEBUG", "INFO", "WARNING", "ERROR")
  - `format`: string, default: "%(asctime)s - %(levelname)s - %(message)s"
  - `file`: string, optional (if provided, logs are written to this file)

Example:

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file: SpaCE-GLUE.log
```