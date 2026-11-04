# Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models

This repository contains the official implementation of:

**Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models**

SeungWon Seo\*, DongHeun Han\*, SeongRae Noh, and HyeongYeop Kang

[Project Page](https://ssw03270.github.io/BABAinWonderland_NeurIPS-26_Page/) | [arXiv](https://arxiv.org/abs/2605.16725)

> If you encounter a bug or have trouble running the code, please open an issue in this repository.
> If I miss the notification or your question is urgent, please contact me at [ssw03270@korea.ac.kr](mailto:ssw03270@korea.ac.kr). I will review it and respond as soon as possible.

## Overview

This repository includes **Alice**, an online self-supervised dynamics discovery system for executable world models in **Baba in Wonderland**.

Alice learns a persistent Python world model from transition evidence. When a candidate program update explains a new transition but breaks previously explained transitions, Alice treats the rejected update as structural signal: the preservation conflict refines hypothesis classes, and those classes are reused both for compact program-update evidence and for update-guided exploration.

## Repository Layout

```text
BABAinWonderland_NeurIPS-26/
  baba_in_wonderland/      Baba in Wonderland environment implementation
  configs/                 experiment, environment, and LLM configs
  custom_maps/             custom map JSON files used by env/baba_custom_ascii_original
  scripts/                 run, data, evaluation, and map editor entrypoints
  src/                     Alice, program discovery, agents, map editor backend, and utilities
  test_dataset/            archived solution and coverage transition datasets
  webui/                   map editor UI assets
```

## Setup

Create the conda environment named `alice` with Python 3.10:

```bash
git clone https://github.com/ssw03270/BABAinWonderland_NeurIPS-26.git
cd BABAinWonderland_NeurIPS-26
conda env create -f environment.yaml
conda activate alice
```

To update an existing environment after dependency changes:

```bash
conda env update -f environment.yaml --prune
```

The LLM client calls OpenAI and Gemini through direct HTTP requests, so no provider SDK is required.

## LLM Credentials

Set API keys through environment variables.

On Linux/macOS:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
```

or create `configs/llm_credentials.yaml` from the example:

```yaml
api_keys:
  openai_api_key: "..."
  gemini_api_key: "..."
```

The active provider and model are selected in `configs/llm_config.yaml`:

```yaml
inference:
  provider: "openai"  # openai | gemini
  openai_model: "gpt-5.4"
  openai_reasoning_effort: "medium"
```

The default is the main model used in the paper (GPT-5.4 with medium reasoning effort).

## Running Alice

The Python commands below are written as single lines so they can be pasted into either Bash or Windows PowerShell.

Run online discovery with the default Baba in Wonderland custom-map split:

```bash
python scripts/run_discovery.py --config configs/experiment_config_online.yaml --llm-config configs/llm_config.yaml --env-config configs/env_config.yaml
```

Progress is printed to the terminal. Each run writes its outputs, including program versions and resolved config snapshots, to `experiments/<name>_seed-<seed>_<timestamp>/`. Use `--seed` to override the experiment seed.

The default settings follow the paper. A run stops when 100 candidate program updates have been accepted or rejected (`experiment.max_total_llm_calls`), when 1,000,000 environment steps are used (`experiment.max_global_steps`), or when no update is incorporated for 300,000 steps (`python_program.saturation_global_step_window`). The explorer settings are in `src/agents/configs/graph_contrastive_agent.yaml`.

Run the fixed-data/offline configuration:

Offline discovery consumes manual-transition artifacts. Export the archived solution dataset first:

```bash
python scripts/data_collect/export_offline_dataset_as_manual_transitions.py --dataset-kind solution --split all_train --world-setting wonderland --experiment-config configs/experiment_config_offline.yaml --output-dir artifacts/manual_transitions/env__baba_custom_ascii_original --overwrite
```

Then run the offline configuration:

```bash
python scripts/run_discovery.py --config configs/experiment_config_offline.yaml
```

## Environment and Maps

The default environment is configured in `configs/env_config.yaml`:

```yaml
environment:
  name: "env/baba_custom_ascii_original"
  kwargs:
    scenario_split: "all_train"
    split_manifest_path: "configs/custom_map_splits_dot_test.yaml"
```

Custom maps live directly under `custom_maps/` as `custom_map_XX.json`. The train/test scenario split is defined in `configs/custom_map_splits_dot_test.yaml`.

The paper evaluates 32 regular levels and 8 held-out extra levels. Online runs collect interaction data on the regular-level split. Offline runs train from archived regular-level solution transitions and evaluate learned programs against held-out extra-level transitions.

This repository includes archived transition/state arrays for both `test_dataset/solution` and `test_dataset/coverage`. Generated metadata files such as `batch_summary.json`, per-map summary JSON files, and `heuristic_dynamics_discovery.json` are omitted to keep the repository small; the loaders scan `*_transitions.jsonl` and `*_states.npz` directly. To regenerate the heuristic class metadata used for class-reduced / Balanced Acc. evaluation, run:

```bash
python scripts/data_collect/heuristic_dynamics_discovery.py --dataset-root test_dataset/solution --output-json test_dataset/solution/heuristic_dynamics_discovery.json
python scripts/data_collect/heuristic_dynamics_discovery.py --dataset-root test_dataset/coverage --output-json test_dataset/coverage/heuristic_dynamics_discovery.json
```

To regenerate coverage transitions locally, run:

```bash
python scripts/data_collect/data_coverage.py --difficulty original --scenario-split all_test --split-manifest configs/custom_map_splits_dot_test.yaml --output-root test_dataset/coverage --workers 8
```

The same script can regenerate other splits by changing `--scenario-split`, for example `all_train`.

## Assets and Permission

This benchmark uses **Baba Is You** as the underlying game setting. Baba in Wonderland preserves the simulator dynamics, action space, rule parsing, sprites, and level/map structure while remapping only the observable rule-property labels used in the agent's state representation.

The original Baba Is You game assets and level/map data are credited to the original game. We obtained permission from the game's developer to use these assets and map data for this research package. This repository does not grant separate rights to reuse the original game assets or map data outside the permitted research context; preserve this attribution and usage notice when sharing derived packages.

The environment code in `baba_in_wonderland/` is adapted from [baba-is-ai](https://github.com/nacloos/baba-is-ai) and keeps its MIT license in `baba_in_wonderland/LICENSE`.

## Useful Tools

Launch the custom map editor:

```bash
python scripts/edit_baba_map.py --difficulty original
```

Evaluate a learned program version on an archived offline dataset:

```bash
python scripts/data_collect/evaluate_offline_dataset.py --dataset-root test_dataset/solution --experiment-dir experiments/<run_dir>
```

The evaluator reads the Wonderland word aliases from the run's `config_snapshot/experiment_config.resolved.yaml`, so archived states are shown to the program in the vocabulary it was trained on. With `--program-file`, pass `--experiment-config configs/experiment_config_online.yaml` instead.

## Paper Context

The accompanying paper describes:

- **Baba in Wonderland**, a prior-misaligned variant of Baba Is You that preserves simulator dynamics while remapping rule-property labels to unrelated words.
- **Alice**, a closed-loop executable world-model learner that uses failed program updates to refine hypothesis classes.
- **Update-guided exploration**, where hypothesis classes shape the learned state-action embedding and frontier scoring.

## Citation

If you find this work useful in your research, please cite the paper as listed on the [project page](https://ssw03270.github.io/BABAinWonderland_NeurIPS-26_Page/).
