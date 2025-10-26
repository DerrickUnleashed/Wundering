#!/usr/bin/env python3
"""
Analyze Hyperparameter Search Results

Identifies patterns and creates refined search space for Phase 2.

Usage: python3 analyze_results.py <results_csv>
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results(results_file):
    """Analyze search results and provide insights."""
    df = pd.read_csv(results_file)

    print("="*70)
    print("HYPERPARAMETER SEARCH ANALYSIS")
    print("="*70)
    print(f"Total configurations: {len(df)}")
    print(f"Best validation R²: {df['best_val_r2'].max():.6f}")
    print(f"Mean validation R²: {df['best_val_r2'].mean():.6f}")
    print(f"Median overfitting gap: {df['overfitting_gap'].median():.6f}")
    print()

    # Top 10 configurations
    print("Top 10 Configurations:")
    print("="*70)
    top10 = df.nlargest(10, 'best_val_r2')
    display_cols = ['config_id', 'best_val_r2', 'overfitting_gap', 'lookback',
                   'hidden_size', 'num_layers', 'dropout', 'bidirectional',
                   'use_gru', 'batch_size', 'lr', 'weight_decay', 'n_parameters']
    print(top10[display_cols].to_string(index=False))
    print()

    # Analyze parameter importance
    print("Parameter Analysis:")
    print("="*70)

    # Continuous parameters - correlation with val R²
    continuous_params = ['lookback', 'hidden_size', 'num_layers', 'dropout',
                        'batch_size', 'lr', 'weight_decay']
    correlations = {}
    for param in continuous_params:
        corr = df[param].corr(df['best_val_r2'])
        correlations[param] = corr

    print("\nCorrelation with Validation R²:")
    for param, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {param:20s}: {corr:+.4f}")

    # Categorical parameters - mean R² by category
    categorical_params = ['bidirectional', 'use_gru', 'fc_activation',
                         'use_batch_norm', 'optimizer']

    print("\nMean Val R² by Category:")
    for param in categorical_params:
        print(f"\n  {param}:")
        means = df.groupby(param)['best_val_r2'].agg(['mean', 'count'])
        for idx, row in means.iterrows():
            print(f"    {str(idx):10s}: {row['mean']:.6f} (n={int(row['count'])})")

    # Overfitting analysis
    print("\n" + "="*70)
    print("Overfitting Analysis:")
    print("="*70)

    # Low overfitting configs (gap < 0.10)
    low_overfit = df[df['overfitting_gap'] < 0.10]
    print(f"Configs with gap < 0.10: {len(low_overfit)}/{len(df)}")
    if len(low_overfit) > 0:
        print(f"Best val R² among low-overfit configs: {low_overfit['best_val_r2'].max():.6f}")
        print("\nTop 5 low-overfitting configs:")
        print(low_overfit.nlargest(5, 'best_val_r2')[display_cols].to_string(index=False))

    # Parameter ranges for top performers
    print("\n" + "="*70)
    print("Recommended Ranges for Phase 2 (based on top 10):")
    print("="*70)

    recommendations = {}
    for param in continuous_params:
        values = top10[param].values
        recommendations[param] = {
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'median': np.median(values)
        }
        print(f"{param:20s}: [{values.min()}, {values.max()}], "
              f"median={np.median(values)}")

    # Most common categorical values in top 10
    print("\nMost common categorical values in top 10:")
    for param in categorical_params:
        value_counts = top10[param].value_counts()
        most_common = value_counts.index[0]
        count = value_counts.iloc[0]
        print(f"  {param:20s}: {most_common} ({count}/10)")

    # Generate refined search space
    print("\n" + "="*70)
    print("Suggested Refined Search Space:")
    print("="*70)

    # Create focused ranges
    refined_space = {}

    for param in continuous_params:
        top_values = top10[param].values
        mean_val = top_values.mean()

        if param == 'lookback':
            # Integer, create range around mean
            refined_space[param] = sorted(list(set([
                max(5, int(mean_val - 10)),
                max(5, int(mean_val - 5)),
                int(mean_val),
                int(mean_val + 5),
                int(mean_val + 10)
            ])))
        elif param in ['hidden_size', 'batch_size']:
            # Powers of 2, create range
            refined_space[param] = sorted(list(set([
                max(32, int(2 ** (np.log2(mean_val) - 1))),
                int(2 ** np.log2(mean_val)),
                int(2 ** (np.log2(mean_val) + 1))
            ])))
        elif param == 'num_layers':
            refined_space[param] = sorted(list(set([
                max(1, int(mean_val - 1)),
                int(mean_val),
                int(mean_val + 1)
            ])))
        elif param in ['dropout', 'weight_decay']:
            # Log scale for regularization
            refined_space[param] = sorted(list(set([
                max(0, mean_val * 0.5),
                mean_val,
                mean_val * 2.0
            ])))
        elif param == 'lr':
            # Log scale for learning rate
            refined_space[param] = sorted(list(set([
                mean_val * 0.5,
                mean_val,
                mean_val * 2.0
            ])))

    for param, values in refined_space.items():
        print(f"{param:20s}: {values}")

    # Categorical - use most common from top 10
    for param in categorical_params:
        value_counts = top10[param].value_counts()
        # Take top 2 values
        top_values = value_counts.head(2).index.tolist()
        refined_space[param] = top_values
        print(f"{param:20s}: {top_values}")

    # Save refined space to JSON
    import json
    refined_file = Path(results_file).parent / 'refined_search_space.json'
    with open(refined_file, 'w') as f:
        # Convert numpy types to native Python types
        clean_space = {}
        for k, v in refined_space.items():
            if isinstance(v, list):
                clean_space[k] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
            else:
                clean_space[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
        json.dump(clean_space, f, indent=2)

    print(f"\nRefined search space saved to: {refined_file}")

    # Estimated configs for Phase 2
    n_configs = 1
    for v in refined_space.values():
        n_configs *= len(v)
    print(f"Estimated configurations for grid search: {n_configs}")
    print(f"Consider random sampling ~50-100 configs if this is too large")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_results.py <results_csv>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    analyze_results(results_file)
