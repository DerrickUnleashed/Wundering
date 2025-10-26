#!/usr/bin/env python3
"""
Training Monitor

Monitors LSTM training progress by checking:
- Process status
- Model file updates
- Checkpoint contents
- Training metrics

Usage: python3 monitor_training.py [PID]
"""

import sys
import time
import os
import torch
import json
from datetime import datetime
from pathlib import Path


def get_process_info(pid):
    """Get process information."""
    try:
        import psutil
        process = psutil.Process(pid)
        return {
            'running': process.is_running(),
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'status': process.status()
        }
    except:
        # Fallback to ps command
        import subprocess
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'state=,pcpu=,pmem='],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            return {
                'running': True,
                'cpu_percent': float(parts[1]) if len(parts) > 1 else 0,
                'memory_mb': 0,
                'status': parts[0] if parts else 'Unknown'
            }
        return {'running': False}


def get_checkpoint_info(checkpoint_path):
    """Extract info from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'val_loss': checkpoint.get('val_loss', 'Unknown'),
            'val_r2': checkpoint.get('val_r2', 'Unknown'),
            'train_r2': checkpoint.get('train_r2', 'Unknown'),
            'exists': True
        }
    except Exception as e:
        return {'exists': False, 'error': str(e)}


def get_file_age(filepath):
    """Get file modification time."""
    try:
        mtime = os.path.getmtime(filepath)
        age_seconds = time.time() - mtime
        return age_seconds
    except:
        return None


def format_time(seconds):
    """Format seconds into readable time."""
    if seconds is None:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def monitor_training(pid=None, interval=5):
    """Monitor training progress."""

    # Files to monitor
    checkpoint_file = 'models/lstm_full_best.pt'
    results_file = 'models/lstm_full_results.json'
    history_file = 'models/lstm_full_history.csv'

    print("="*70)
    print("LSTM TRAINING MONITOR")
    print("="*70)

    if pid:
        print(f"Monitoring PID: {pid}")
    print(f"Refresh interval: {interval}s")
    print(f"Press Ctrl+C to stop monitoring\n")

    iteration = 0
    last_checkpoint_time = None

    try:
        while True:
            iteration += 1
            os.system('clear' if os.name != 'nt' else 'cls')

            print("="*70)
            print(f"LSTM TRAINING MONITOR - Update #{iteration}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)

            # Process status
            if pid:
                proc_info = get_process_info(pid)
                print(f"\n📊 Process Status (PID {pid}):")
                if proc_info['running']:
                    print(f"  Status: ✓ RUNNING")
                    print(f"  CPU: {proc_info['cpu_percent']:.1f}%")
                    if proc_info['memory_mb'] > 0:
                        print(f"  Memory: {proc_info['memory_mb']:.1f} MB")
                else:
                    print(f"  Status: ✗ NOT RUNNING")
                    print(f"  Training may have completed or crashed!")

            # Checkpoint status
            print(f"\n📁 Model Files:")
            if os.path.exists(checkpoint_file):
                checkpoint_age = get_file_age(checkpoint_file)
                checkpoint_size = os.path.getsize(checkpoint_file) / 1024 / 1024

                print(f"  Checkpoint: ✓ EXISTS")
                print(f"    Size: {checkpoint_size:.2f} MB")
                print(f"    Last updated: {format_time(checkpoint_age)} ago")

                # Check if checkpoint is stale
                if checkpoint_age and checkpoint_age > 300:  # 5 minutes
                    print(f"    ⚠️  WARNING: No update in {format_time(checkpoint_age)}")

                # Try to read checkpoint info
                checkpoint_info = get_checkpoint_info(checkpoint_file)
                if checkpoint_info['exists']:
                    print(f"    Epoch: {checkpoint_info['epoch']}")
                    if isinstance(checkpoint_info['val_loss'], float):
                        print(f"    Val Loss: {checkpoint_info['val_loss']:.6f}")
                    if isinstance(checkpoint_info.get('val_r2'), float):
                        print(f"    Val R²: {checkpoint_info['val_r2']:.6f}")
                    if isinstance(checkpoint_info.get('train_r2'), float):
                        print(f"    Train R²: {checkpoint_info['train_r2']:.6f}")
            else:
                print(f"  Checkpoint: ✗ NOT CREATED YET")

            # Results file
            if os.path.exists(results_file):
                print(f"\n✅ Training Complete!")
                with open(results_file, 'r') as f:
                    results = json.load(f)
                if 'best_val_r2' in results:
                    print(f"  Best val R²: {results.get('best_val_r2', 'N/A'):.6f}")
                if 'best_val_loss' in results:
                    print(f"  Best val loss: {results.get('best_val_loss', 'N/A'):.6f}")
                print(f"  Total epochs: {results.get('total_epochs', 'N/A')}")
                print(f"  Training time: {results.get('training_time_minutes', 'N/A'):.1f} min")
                print(f"\n🎉 Training finished successfully!")
                break
            else:
                print(f"  Results: ⏳ Training in progress...")

            # History file
            if os.path.exists(history_file):
                import pandas as pd
                history = pd.read_csv(history_file)
                latest = history.iloc[-1]
                print(f"\n📈 Latest Metrics (Epoch {len(history)}):")
                print(f"  Train Loss: {latest['train_loss']:.6f}")
                print(f"  Val Loss: {latest['val_loss']:.6f}")
                if 'train_r2' in history.columns:
                    print(f"  Train R²: {latest['train_r2']:.6f}")
                if 'val_r2' in history.columns:
                    print(f"  Val R²: {latest['val_r2']:.6f}")
                print(f"  Learning Rate: {latest['lr']:.6f}")

            print(f"\n⏱️  Next update in {interval}s...")
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n✋ Monitoring stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    monitor_training(pid, interval)
