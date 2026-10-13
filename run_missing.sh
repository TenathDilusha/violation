#!/usr/bin/env bash
# Run every method on the articles that have no answer yet (e.g. newly added violations).
# The runners skip articles that already have an answer and retry errors, so this can be
# re-run until nothing is left.
#
#   ./run_missing.sh openai|gemini [extra args, e.g. --workers 3]
#
# openai: the 3 tasks × zero_shot / few_shot / per_rule, 3 repeats; rule detection
# zero_shot_v2 / v3, 3 repeats; the 3 tasks with the Sinhala code of conduct, 3 repeats;
# and the two-step pipelines.
# gemini: the 3 tasks × zero_shot / few_shot / per_rule, one run.
set -uo pipefail
cd "$(dirname "$0")"

provider="$1"
shift
extra=("$@")
py=.venv/bin/python

run() {
  echo "=== $*"
  "$py" "$@" --negatives 30 "${extra[@]}" || echo "!!! failed: $*"
}

if [[ "$provider" == "openai" ]]; then
  for r in 1 2 3; do
    for task in binary_detection rule_detection span_detection; do
      for prompt in zero_shot few_shot per_rule; do
        run "$task/run_openai.py" --prompt "$prompt" --repeat "$r"
        run "$task/run_openai.py" --prompt "$prompt" --repeat "$r" --code-lang si --lang si
      done
    done
    for prompt in zero_shot_v2 zero_shot_v3; do
      run rule_detection/run_openai.py --prompt "$prompt" --repeat "$r"
    done
  done
  run two_step/run.py
elif [[ "$provider" == "gemini" ]]; then
  for task in binary_detection rule_detection span_detection; do
    for prompt in zero_shot few_shot per_rule; do
      run "$task/run_gemini.py" --prompt "$prompt"
    done
  done
else
  echo "usage: $0 openai|gemini [args]" >&2
  exit 1
fi
