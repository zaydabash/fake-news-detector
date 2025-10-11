#!/usr/bin/env python3
"""Script to collect results from all trained niches and generate summary tables."""

import json
import os
import pandas as pd
from pathlib import Path


def collect_results():
    """Collect results from all niche summary files."""
    results = []
    
    # Find all summary.json files
    artifacts_dir = Path("artifacts")
    for niche_dir in artifacts_dir.iterdir():
        if niche_dir.is_dir() and niche_dir.name != "results":
            summary_file = niche_dir / "reports" / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                # Extract data for CSV
                samples_per_class = summary["samples_per_class"]
                labels = list(samples_per_class.keys())
                pos_count = samples_per_class[labels[0]]
                neg_count = samples_per_class[labels[1]]
                
                results.append({
                    "niche": summary["niche"],
                    "test_macro_f1": summary["test_macro_f1"],
                    "test_accuracy": summary["test_accuracy"],
                    "pos_count": pos_count,
                    "neg_count": neg_count,
                    "total_samples": pos_count + neg_count
                })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Create results directory
    results_dir = Path("artifacts/results")
    results_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_path = results_dir / "results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved results table to {csv_path}")
    
    # Create Markdown table
    md_path = results_dir / "results_table.md"
    with open(md_path, 'w') as f:
        f.write("## Model Performance Results\n\n")
        f.write("| Niche | Macro F1 | Accuracy | Samples |\n")
        f.write("|-------|----------|----------|----------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['niche']} | {row['test_macro_f1']:.3f} | {row['test_accuracy']:.3f} | {row['total_samples']} |\n")
        
        f.write(f"\n**Total:** {df['total_samples'].sum()} samples across {len(df)} niches\n")
    
    print(f"✅ Saved Markdown table to {md_path}")
    print(f"\n📊 Results Summary:")
    print(df.to_string(index=False))
    
    return df


if __name__ == "__main__":
    collect_results()
