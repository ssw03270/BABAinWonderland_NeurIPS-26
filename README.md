# Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models

This repository contains the official implementation of:

**Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models**

SeungWon Seo\*, DongHeun Han\*, SeongRae Noh, and HyeongYeop Kang

Paper Link: [arXiv](https://arxiv.org/abs/2605.16725)

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
  scripts/                 run, evaluation, player, and editor entrypoints
  src/                     Alice, program discovery, agents, web UI, and utilities
  test_dataset/            archived solution and coverage transition datasets
  webui/                   browser UI assets
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
```

## Running Alice

The Python commands below are written as single lines so they can be pasted into either Bash or Windows PowerShell.

Run online discovery with the default Baba in Wonderland custom-map split:

```bash
python scripts/run_discovery.py --config configs/experiment_config_online.yaml --llm-config configs/llm_config.yaml --env-config configs/env_config.yaml
```

For a headless run without the dashboard:

```bash
python scripts/run_discovery.py --config configs/experiment_config_online.yaml --headless
```

Run the fixed-data/offline configuration:

Offline discovery consumes manual-transition artifacts. Export the archived solution dataset first:

```bash
python scripts/data_collect/export_offline_dataset_as_manual_transitions.py --dataset-kind solution --split all_train --world-setting wonderland --experiment-config configs/experiment_config_offline.yaml --output-dir artifacts/manual_transitions/env__baba_custom_ascii_original --overwrite
```

Then run the offline configuration:

```bash
python scripts/run_discovery.py --config configs/experiment_config_offline.yaml
```

To run offline discovery without the dashboard, add `--headless`:

```bash
python scripts/run_discovery.py --config configs/experiment_config_offline.yaml --headless
```

The discovery dashboard server is shared across runs. If an online run is already using the default dashboard port, the offline run attaches to the same dashboard with a separate `run_id`.

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

This upload package includes archived transition/state arrays for both `test_dataset/solution` and `test_dataset/coverage`. Generated metadata files such as `batch_summary.json`, per-map summary JSON files, and `heuristic_dynamics_discovery.json` are omitted to keep the package within supplement upload limits; the loaders scan `*_transitions.jsonl` and `*_states.npz` directly. To regenerate the heuristic class metadata used for class-reduced / Balanced Acc. evaluation, run:

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

## Useful Tools

Launch the browser-based manual player:

```bash
python scripts/play_baba_env.py --env env/baba_custom_ascii_original
```

Launch the custom map editor:

```bash
python scripts/edit_baba_map.py --difficulty original
```

Evaluate a learned program version on an archived offline dataset:

```bash
python scripts/data_collect/evaluate_offline_dataset.py --help
```

## Paper Context

The accompanying paper describes:

- **Baba in Wonderland**, a prior-misaligned variant of Baba Is You that preserves simulator dynamics while remapping rule-property labels to unrelated words.
- **Alice**, a closed-loop executable world-model learner that uses failed program updates to refine hypothesis classes.
- **Update-guided exploration**, where hypothesis classes shape the learned state-action embedding and frontier scoring.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{seo2026baba,
  title={Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models},
  author={SeungWon Seo and DongHeun Han and SeongRae Noh and HyeongYeop Kang},
  journal={arXiv preprint arXiv:2605.16725},
  year={2026},
  url={https://arxiv.org/abs/2605.16725}
}
```
