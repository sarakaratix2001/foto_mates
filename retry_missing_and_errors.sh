#!/usr/bin/env bash
set -eu
cd '/mnt/c/Users/User/Desktop/oshe/foto mates'
run_dir="reports/retry_missing_errors_$(date +%Y%m%d_%H%M%S)"
exec /home/sara_karati/venv/bin/python3 -u audit_readings.py \
  --db ':memory:' read \
  --files-csv reports/reconciliation_20261026/retry_ready.csv \
  --month 2026-09 --workers 32 \
  --verbose --progress 100 --report-every 0 \
  --no-word --word-dir "$run_dir"
