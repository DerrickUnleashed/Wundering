# Hyperparameter Tuning Status

## Current Status
**Phase 1 - Broad Search:** RUNNING (PID 60470) 🚀 GPU-ENABLED
- Started: 2025-10-18 15:26:22
- Device: **MPS (Apple Metal GPU)**
- Configurations: 50 random samples
- Data: 25% (103 train / 26 val sequences)
- Max epochs: 10, Patience: 5
- Estimated time: **1-2 hours** (GPU-accelerated!)
- Log file: `tune_broad.log` (with epoch-by-epoch details)
- Output: `models/broad_search_results_YYYYMMDD_HHMMSS.csv`

**Enhanced Logging:**
- Shows config details before each run
- Epoch-by-epoch Val R², Train R², and overfitting gap
- Clear visual markers (✓) for improvements

## Progress Tracking
Check progress:
```bash
# View latest results
tail -20 tune_broad.log

# Check checkpoint (saved every 10 configs)
ls -lh models/*checkpoint.csv

# Monitor progress
watch -n 60 'tail -5 tune_broad.log'
```

## Phase Overview

### Phase 0: Test ✅ COMPLETE
- Tested on tiny dataset (8 train, 2 val sequences)
- 5 configurations, 3 epochs each
- All systems working correctly
- Time: 1.3 minutes

### Phase 1: Broad Search 🔄 IN PROGRESS
**Updated Settings:**
- Configurations: 50 (reduced from 100)
- Data: 25% (reduced from 50%)
- Estimated time: 3-4 hours

**17 Hyperparameters:**
1. Lookback: [5, 10, 15, 20, 30, 50]
2. Hidden size: [32, 64, 96, 128, 192, 256]
3. Num layers: [1, 2, 3, 4]
4. Dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
5. Bidirectional: [False, True]
6. Use GRU: [False, True]
7. FC num layers: [1, 2, 3]
8. FC hidden dims: [64, 128, 256]
9. FC activation: ['relu', 'tanh', 'gelu']
10. FC dropout: [0.0, 0.2, 0.3, 0.4]
11. Use batch norm: [False, True]
12. Batch size: [32, 64, 128, 256]
13. Learning rate: [0.0001, 0.0005, 0.001, 0.002, 0.005]
14. Weight decay: [0, 0.0001, 0.001, 0.01]
15. Gradient clip: [None, 0.5, 1.0, 5.0]
16. Optimizer: ['adam', 'adamw']
17. LR patience: [3, 5, 7, 10]

**Search Strategy:** Random sampling (100 configs)

### Phase 2: Refined Search ⏳ PENDING
- Will analyze Phase 1 results
- Create focused grid around top 5 configurations
- ~50-60 configurations
- Max epochs: 25, Patience: 7
- Estimated time: 3-4 hours

### Phase 3: Final Training ⏳ PENDING

**Phase 3a: 50% Data Model**
- Train with best hyperparameters
- Max 30 epochs, patience 10
- Create submission → `wunderfund/submissions/lstm_tuned_50pct.zip`
- Time: ~30 min

**Phase 3b: 100% Data Model**
- Same hyperparameters, full dataset
- Max 30 epochs, patience 10
- Create submission → `wunderfund/submissions/lstm_tuned_100pct.zip`
- Time: ~45 min

## Key Metrics to Track
- **best_val_r2**: Validation R² (higher is better)
- **overfitting_gap**: train_r2 - val_r2 (lower is better, target < 0.10)
- **n_parameters**: Model size
- **training_time_sec**: Time per configuration

## Baseline Performance
- EMA baseline: 0.232 R² (test set)
- LSTM tiny: 0.205 R² (test set)
- **LSTM full (R² optimized): 0.367 R²** (test set) ← CURRENT BEST
  - Val R²: 0.350
  - Overfitting gap: 0.092 (train: 0.442, val: 0.350 at final epoch)
  - Gap at best epoch: 0.047 (train: 0.397, val: 0.350 at epoch 5)

## Goal
Beat current best of 0.367 R² by:
1. Reducing overfitting (gap < 0.05 at best epoch)
2. Improving generalization through better architecture
3. Finding optimal regularization balance

## Next Steps
1. Wait for Phase 1 to complete (~6-8 hours)
2. Analyze results to identify patterns
3. Create refined search script
4. Run Phase 2 refined search
5. Train final models and submit
