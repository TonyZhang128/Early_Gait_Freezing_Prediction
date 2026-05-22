#!/bin/bash
# Shared utilities for training scripts

activate_venv() {
    local path="${1:-.venv/Scripts/activate}"
    [ -f "$path" ] && source "$path"
}

run_train() {
    local label="$1" cmd="$2" log="$3"
    echo -e "\n===== $label =====\n$cmd\n"
    PYTHONUNBUFFERED=1 eval "$cmd" 2>&1 | tee "$log"
}
