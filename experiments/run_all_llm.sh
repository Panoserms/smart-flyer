#!/bin/zsh
# All Gemma runs of the study, in order. Every step is resumable: re-running skips finished calls.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
stamp() { echo "=== $(date '+%H:%M:%S') $1"; }
stamp "prompt selection"; $PY experiments/select_prompt.py
for S in t1 t2 t3; do
  for C in plain informed; do
    stamp "triage $S $C"; $PY experiments/run_triage.py --set $S --condition $C --variant auto
  done
done
stamp "robustness"; $PY experiments/run_robustness.py --variant auto
stamp "copy"; $PY experiments/run_copy.py
stamp "rating sheets"; $PY experiments/make_rating_sheet.py
stamp "done"
