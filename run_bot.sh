#!/usr/bin/env bash
# Box fallback scheduler (only if GitHub Actions isn't used). Cron line:
#   7,27,47 * * * * /workspace/money/metaculus/template/run_bot.sh
cd "$(dirname "$0")" || exit 1
exec 9>/tmp/metaculus_bot.lock; flock -n 9 || exit 0   # never run twice at once
mkdir -p logs
source .venv/bin/activate
python main_free.py --mode tournament >> "logs/$(date +%F).log" 2>&1
