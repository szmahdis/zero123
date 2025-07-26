#!/bin/bash

# Sequential Zero123 Training Script with Text Logging
# Runs vanilla experiment followed by plucker experiment

set -e  # Exit on any error

# Create logs directory if it doesn't exist
mkdir -p experiment_logs

# Get current timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Main log file for the entire run
MAIN_LOG="experiment_logs/sequential_run_${TIMESTAMP}.txt"

echo "=== Starting Sequential Zero123 Training at $(date) ===" | tee "$MAIN_LOG"
echo "All outputs will be saved to experiment_logs/ directory" | tee -a "$MAIN_LOG"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log_message "=== EXPERIMENT 1: VANILLA MODEL ==="
log_message "Starting vanilla experiment..."

# Run vanilla experiment and save all output
VANILLA_LOG="experiment_logs/vanilla_experiment_${TIMESTAMP}.txt"
log_message "Vanilla experiment output will be saved to: $VANILLA_LOG"

python main.py --base configs/sd-objaverse-vanilla-c_concat-256.yaml --train --name vanilla_experiment 2>&1 | tee "$VANILLA_LOG"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_message "✅ Vanilla experiment completed successfully"
else
    log_message "❌ Vanilla experiment failed - stopping execution"
    exit 1
fi

log_message "=== Waiting 30 seconds before next experiment ==="
sleep 30

log_message "=== EXPERIMENT 2: PLUCKER MODEL ==="
log_message "Starting plucker experiment..."

# Run plucker experiment and save all output
PLUCKER_LOG="experiment_logs/plucker_experiment_${TIMESTAMP}.txt"
log_message "Plucker experiment output will be saved to: $PLUCKER_LOG"

python main.py --base configs/sd-objaverse-plucker-c_concat-256.yaml --train --name plucker_experiment 2>&1 | tee "$PLUCKER_LOG"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_message "✅ Plucker experiment completed successfully"
else
    log_message "❌ Plucker experiment failed"
    exit 1
fi

log_message "=== ALL EXPERIMENTS COMPLETED SUCCESSFULLY! ==="
log_message "Total runtime: $SECONDS seconds"
log_message "Log files created:"
log_message "  - Main log: $MAIN_LOG"
log_message "  - Vanilla log: $VANILLA_LOG"  
log_message "  - Plucker log: $PLUCKER_LOG"

echo "=== SUMMARY ===" | tee -a "$MAIN_LOG"
echo "Vanilla log: $VANILLA_LOG" | tee -a "$MAIN_LOG"
echo "Plucker log: $PLUCKER_LOG" | tee -a "$MAIN_LOG"
echo "Main log: $MAIN_LOG" | tee -a "$MAIN_LOG"