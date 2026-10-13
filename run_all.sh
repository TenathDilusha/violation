#!/usr/bin/env bash
# Run every task (binary_detection, rule_detection, span_detection) × prompt (zero_shot, few_shot, per_rule) for one provider.
#
#   ./run_all.sh openai|gemini|sinllama [extra args, e.g. --force --workers 6 --repeat 2 --code-lang si --lang si]
#
# Uses --negatives 30 (30 non-violation articles per language). SinLlama runs in
# anish_project's venv and only on Sinhala articles.
set -euo pipefail
cd "$(dirname "$0")"

provider="$1"
shift
if [[ "$provider" == "sinllama" ]]; then
  python=(env PYTHONPATH=/home/dilusha/anish_project/.python_pkgs /home/dilusha/anish_project/venv/bin/python)
else
  python=(.venv/bin/python)
fi

for task in binary_detection rule_detection span_detection; do
  for prompt in zero_shot few_shot per_rule; do
    echo "=== $task · $prompt · $provider"
    "${python[@]}" "$task/run_$provider.py" --prompt "$prompt" --negatives 30 "$@"
  done
done
