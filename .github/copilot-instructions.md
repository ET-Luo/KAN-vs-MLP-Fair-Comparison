# Copilot Agent Instructions

## Environment Activation
When using Python in agent mode (e.g., running scripts or installing dependencies), always activate the conda environment first:

```
source ~/miniconda3/etc/profile.d/conda.sh
conda activate model
```

## Notes
- Prefer commands that run from the project root.
- Keep actions idempotent where possible.
