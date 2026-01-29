#!/bin/sh
# POSIX-compatible project initialization script

set -eu

# Directories to create
DIRS="\
src \
src/models \
src/data \
src/training \
src/utils \
experiments \
data/raw \
data/processed \
models \
results/logs \
results/metrics \
results/figures \
configs \
scripts \
logs"

# Create directories
for d in $DIRS; do
    mkdir -p "$d"
done

# Create placeholder files only if they do not exist
create_file_if_missing() {
    file_path="$1"
    shift
    if [ ! -f "$file_path" ]; then
        # shellcheck disable=SC2059
        printf "%s" "$*" > "$file_path"
    fi
}

create_file_if_missing "src/__init__.py" "# Package initializer\n"
create_file_if_missing "src/models/__init__.py" "# Models package\n"
create_file_if_missing "src/data/__init__.py" "# Data package\n"
create_file_if_missing "src/training/__init__.py" "# Training package\n"
create_file_if_missing "src/utils/__init__.py" "# Utils package\n"

create_file_if_missing "configs/default.yaml" "dataset:\n  name: \"\"\n  path: \"data/raw\"\n\nmodel:\n  name: \"\"\n  params: {}\n\ntraining:\n  epochs: 0\n  batch_size: 0\n  learning_rate: 0.0\n\nlogging:\n  level: \"INFO\"\n  log_dir: \"logs\"\n"

create_file_if_missing "requirements.txt" "# Add project dependencies here\n"

# Print tree-like summary
printf "Created/verified project structure:\n\n"
printf ".\n"
printf "├── configs\n"
printf "│   └── default.yaml\n"
printf "├── data\n"
printf "│   ├── processed\n"
printf "│   └── raw\n"
printf "├── experiments\n"
printf "├── logs\n"
printf "├── models\n"
printf "├── results\n"
printf "│   ├── figures\n"
printf "│   ├── logs\n"
printf "│   └── metrics\n"
printf "├── scripts\n"
printf "├── src\n"
printf "│   ├── __init__.py\n"
printf "│   ├── data\n"
printf "│   │   └── __init__.py\n"
printf "│   ├── models\n"
printf "│   │   └── __init__.py\n"
printf "│   ├── training\n"
printf "│   │   └── __init__.py\n"
printf "│   └── utils\n"
printf "│       └── __init__.py\n"
printf "└── requirements.txt\n"
