# Repository Guidelines

This repo is a fork of the official openpi (physical-intelligence) codebase, used to
**full fine-tune pi0 / pi0.5 base on custom LeRobot v2.1 datasets** (no offline conversion). The
codebase is **dataset-agnostic**: the pi0 model is generic, and each dataset is supported by a
matched pair of a **data transform** (`src/openpi/policies/`) and a **train/data config**
(`src/openpi/training/config.py`). The model code is upstream and stays untouched.

The shipped example is a **dual-arm UMI** dataset (LeRobot v2.1, `dual_arm`, 30 fps, AV1 video):
per frame a 23-dim absolute EE state/action (per arm `pos3 + quat_wxyz4 + grip1`, + 7 ego dims that
are dropped) plus two wrist cameras. **If your dataset matches that layout, you only add a config;
if it differs, you also write a transform** — both paths are in the [README](README.md)
Fine-Tuning Guide (Steps 0, 0.5).

## Project Structure & Module Organization

- Core Python package: `src/openpi/` — models (`models/`, `models_pytorch/`), policies
  (`policies/`), training (`training/`), shared utils (`shared/`), `transforms.py`.
- Workspace package: `packages/openpi-client/` (client library + runtime, has its own tests).
- Entry points: `scripts/` — `train.py`, `train_pytorch.py`, `serve_policy.py`,
  `compute_norm_stats.py`.
- Examples and platform guides: `examples/`, `docs/`.
- Tests are co-located with code: `src/**/*_test.py`, `packages/**/*_test.py`,
  `scripts/test_*.py`.
- **Custom to this fork:**
  - `src/openpi/policies/umi_dual_arm_policy.py` — quaternion (16-dim) UMI transform.
  - `src/openpi/policies/umi_dual_arm_rot6d_policy.py` — 6D-rotation (20-dim) UMI transform.
  - `UmiDualArmDataConfig` / `UmiDualArmRot6dDataConfig` and the `pi0_umi_dual_arm*` /
    `pi05_umi_dual_arm*` `TrainConfig` blocks in `training/config.py`.
  - `README.md` — the fork's operator-facing fine-tune guide (Config Menu + end-to-end steps;
    folds in the former `FINETUNE.md`).

## Build, Test, and Development Commands

Install deps and dev tools (skip Git LFS smudge; the model weights are pulled on demand):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --dev
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Run the full test suite (matches CI):

```bash
uv run pytest --strict-markers -m "not manual"
```

Lint and format (Ruff is the single source of truth, max line length 120):

```bash
uv run ruff check .
uv run ruff format .
pre-commit run -a
```

Typical fine-tune flow (see `README.md` and the `pi0-umi-finetune` skill for the full story):

```bash
uv run scripts/compute_norm_stats.py --config-name <config>
uv run scripts/train.py <config> --exp-name=<run_name> --fsdp-devices=4 --overwrite
```

## Environment Gotchas (read before running data/train commands)
- **Full fine-tune runs remote on H20,A100 etc**, sharded with `--fsdp-devices`.
- **Norm stats** are namespaced by config `name` under `assets/<name>/<asset_id>/`; give each
  dataset a unique `asset_id`. Stats are computed on the **relativized** state/actions, so they
  must be recomputed whenever the representation or dataset changes.

## Coding Style & Naming Conventions

- Python 3.11+ (`pyproject.toml`). Ruff for lint/format, max line length 120.
- Follow existing import style (single-line imports, sorted per Ruff/isort).
- Prefer descriptive module names; tests follow `<module>_test.py` or `test_<feature>.py`.
- Keep the two UMI policy modules separate (quat vs rot6d) — do not merge them.

## Testing Guidelines

- Framework: `pytest`; `testpaths` covers `src`, `scripts`, `packages`.
- Use `@pytest.mark.manual` only for tests needing hardware/manual execution.
- Keep automated tests deterministic and runnable via `uv run pytest --strict-markers -m "not manual"`.
- For transform changes, verify the SE(3) relativize/absolutize round-trip numerically
  (should close to ~1e-7 through the full Normalize → pad → Unnormalize → absolutize path).

## Commit & Pull Request Guidelines

- Prefer short, imperative titles (`fix ...`, `feat: ...`), under ~70 chars.
- Keep commits focused (one logical change); add context in the body when needed.
- PRs should include a clear title/description, linked issues/discussions when relevant,
  passing `pre-commit`/Ruff/pytest, and repro details/logs for behavior changes.

## Documentation Management

**Core principle**: `.claude/skills/` is the central knowledge base for this repo. Archive any
durable knowledge (workflows, fixes, design decisions, debugging findings) into the most
relevant skill; create a new one with `skill-creator` if none fits. Keep skills current.

- After finishing work, update the relevant skill: workflows/usage for features, a
  "Common Issues" note for bug fixes, best-practice params for perf work, concept/architecture
  notes for design changes.
- **Minimize other docs**: avoid standalone files under `docs/`, per-subdir `README.md`, and
  transient task-completion reports. Scattered docs rot; centralized skills are maintained.
- Doc hierarchy: **AGENTS.md** (this file — overview, env, quick start) → **skills**
  (`.claude/skills/` — subsystem detail, workflows, troubleshooting) → **code comments**
  (implementation detail). The operator-facing fine-tune guide now lives in `README.md`:
  a quick-start that points into the skill for depth.
