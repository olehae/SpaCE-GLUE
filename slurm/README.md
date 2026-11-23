# Running SpaCE-GLUE on the GWDG SCC

These scripts run SpaCE-GLUE on the Scientific Compute Cluster (SCC) of the
GWDG. A GPU job starts a [vLLM](https://docs.vllm.ai) server for a Hugging Face
model and evaluates it through the OpenAI-compatible API.

| File | Purpose |
|------|---------|
| `setup_scc.sh` | One-time setup of the Python environment (run on a login node) |
| `scc_smoke_test.sbatch` | CPU-only smoke test against a stub endpoint, no GPU needed |
| `scc_vllm.sbatch` | GPU job: vLLM server + SpaCE-GLUE evaluation |
| `config.scc.yaml` | SpaCE-GLUE configuration used by both jobs |
| `stub_openai_server.py` | Stub endpoint used by the smoke test |

All commands below are run **from the repository root**.

## 1. Setup (once)

Log in to the SCC and clone the repository. Results and cached datasets are
written into the repository directory, so prefer a location in your project
directory (`$PROJECT`) over `$HOME`.

```bash
git clone https://github.com/olehae/SpaCE-GLUE.git
cd SpaCE-GLUE
bash slurm/setup_scc.sh
```

This creates `.venv/` with Python 3.13 (via the `uv` module) and installs
SpaCE-GLUE into it. The first run takes about ten minutes.

## 2. Smoke test (CPU only)

```bash
sbatch slurm/scc_smoke_test.sbatch
```

The job takes about two minutes on `scc-cpu`. It downloads the datasets into
`data/` and runs inference, scoring and aggregation against a stub endpoint
that always gives the same answer. Check the end of
`slurm-space-glue-smoke-<jobid>.out`:

```text
SMOKE TEST PASSED - results (stub answers, not meaningful) are in results/smoke-test-<jobid>
```

The scores of the smoke test are meaningless. Its purpose is to verify the
setup and to pre-download the datasets, so the GPU job does not spend GPU time
on that.

## 3. Evaluate a model (GPU)

The GPU job runs vLLM from an [Apptainer](https://apptainer.org) image. If your
group already has a vLLM image, point `VLLM_SIF` at it. Otherwise build one from
the official Docker image (about 10 GB, do this in your project directory):

```bash
module load apptainer
apptainer pull vllm-openai.sif docker://vllm/vllm-openai:latest
```

Then submit the job:

```bash
export VLLM_SIF=/path/to/vllm.sif   # not needed if ./vllm-openai.sif exists
sbatch slurm/scc_vllm.sbatch
```

By default this evaluates `Qwen/Qwen2.5-7B-Instruct` on one A100 with the
datasets listed in `slurm/config.scc.yaml`. Settings are changed with
environment variables, either exported before `sbatch` or passed with
`--export`:

```bash
sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct slurm/scc_vllm.sbatch
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face repo id or path to a local model directory |
| `MODEL_NAME` | same as `MODEL` | Name used in the API and in the result file names |
| `CONFIG` | `slurm/config.scc.yaml` | SpaCE-GLUE configuration file |
| `RESULTS_DIR` | `results` | Directory for the results |
| `VLLM_SIF` | `./vllm-openai.sif` | Apptainer image that contains vLLM |
| `VLLM_ARGS` | empty | Extra arguments for `vllm serve`, e.g. `--max-model-len 16384` |
| `HF_HOME` | `$PROJECT/$USER/hf` | Hugging Face cache (models and datasets) |
| `HF_TOKEN` | unset | Hugging Face token, needed for gated models |

To change the datasets, the batch size or the number of runs, copy
`slurm/config.scc.yaml`, edit the copy and pass it with `CONFIG`. Class paths in
the config need the `space_glue.` prefix, e.g. `space_glue.data.SPaRC.SPaRC`.

Slurm options can be overridden on the command line as usual, for example a
bigger GPU request and a longer time limit for a larger model:

```bash
sbatch -G A100:2 --time=24:00:00 --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ slurm/scc_vllm.sbatch
```

### Output

- `slurm-space-glue-<jobid>.out`: job log with the SpaCE-GLUE output
- `slurm-vllm-<jobid>.log`: log of the vLLM server (look here if the model does not start)
- `results/<dataset>_<model>.jsonl`: responses and scores per item
- `results/<dataset>_<model>.summary.json`: aggregated scores

Monitor the job with `squeue --me` and `tail -f slurm-space-glue-<jobid>.out`.

### Resuming

If a job is interrupted (for example by the time limit), submit it again with
the same settings. Items that already have a response are skipped. A dataset
that already has a `.summary.json` is finished and is reported as an error if
it is run again; remove or rename its result files to re-evaluate it.

## Known issues

- **bAbI and SpartQA cannot be downloaded at the moment** and are therefore
  commented out in `config.scc.yaml`. This is not specific to the cluster: the
  bAbI source archive (`thespermwhale.com`) returns 404, and the SpartQA host
  (`www.cse.msu.edu`) serves an incomplete TLS certificate chain. If you have
  `data/babi_data.jsonl` and `data/spartqa_data.jsonl` from an earlier run on
  another machine, copy them into `data/` and enable the two datasets in the
  config (remember to add two entries to `runs`).
- **No internet in a job**: compute nodes only reach the internet with
  `--constraint=inet` (set in both job scripts), which routes http(s) through
  the GWDG proxy.
- **Failed dataset download**: SpaCE-GLUE skips a dataset that fails and
  continues with the next one. The job scripts report this at the end of the log
  and exit with an error. Fix the cause and submit again.
