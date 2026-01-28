#!/usr/bin/env python3
"""
Compare CHAIR-M scores across different experiments/models.
"""

import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_label(csv_path: str, label: str) -> pd.DataFrame:
    """Load CSV and add experiment label."""
    df = pd.read_csv(csv_path)
    df['experiment'] = label
    return df


def compare_experiments(csv_paths: list, labels: list, output_dir: str = "."):
    """
    Compare CHAIR-M scores across multiple experiments.
    
    Args:
        csv_paths: List of CSV file paths
        labels: List of experiment labels
        output_dir: Directory to save comparison results
    """
    # Load all experiments
    dfs = [load_and_label(path, label) for path, label in zip(csv_paths, labels)]
    combined = pd.concat(dfs, ignore_index=True)
    
    # Summary statistics
    summary = combined.groupby('experiment').agg({
        'chair_claim_score': ['mean', 'std', 'median'],
        'chair_sentence_score': ['mean', 'std', 'median'],
        'num_visual_claims': 'mean',
        'num_audio_claims': 'mean',
        'num_hallucinated_claims': 'mean',
    }).round(4)
    
    print("\n" + "="*80)
    print("CHAIR-M Score Comparison")
    print("="*80)
    print(summary)
    print("="*80)
    
    # Save summary
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "chair_m_comparison_summary.csv")
    print(f"\nSaved summary to {output_dir / 'chair_m_comparison_summary.csv'}")
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Claim level scores
    sns.boxplot(data=combined, x='experiment', y='chair_claim_score', ax=axes[0])
    axes[0].set_title('CHAIR-M (Claim Level)')
    axes[0].set_ylabel('Hallucination Score')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Sentence level scores
    sns.boxplot(data=combined, x='experiment', y='chair_sentence_score', ax=axes[1])
    axes[1].set_title('CHAIR-M (Sentence Level)')
    axes[1].set_ylabel('Hallucination Score')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = output_dir / "chair_m_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    
    # Statistical tests
    print("\n" + "="*80)
    print("Pairwise Comparisons (t-test)")
    print("="*80)
    
    from scipy import stats
    
    if len(labels) == 2:
        exp1_data = combined[combined['experiment'] == labels[0]]
        exp2_data = combined[combined['experiment'] == labels[1]]
        
        # Claim level
        t_stat, p_val = stats.ttest_ind(
            exp1_data['chair_claim_score'],
            exp2_data['chair_claim_score']
        )
        print(f"Claim Level: {labels[0]} vs {labels[1]}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
        
        # Sentence level
        t_stat, p_val = stats.ttest_ind(
            exp1_data['chair_sentence_score'],
            exp2_data['chair_sentence_score']
        )
        print(f"Sentence Level: {labels[0]} vs {labels[1]}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare CHAIR-M scores across experiments")
    parser.add_argument("csv_files", nargs="+", help="CSV files to compare")
    parser.add_argument("--labels", nargs="+", default=None, 
                       help="Labels for each experiment (default: use filenames)")
    parser.add_argument("--output-dir", default=".", 
                       help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # Generate labels if not provided
    labels = args.labels or [Path(f).stem for f in args.csv_files]
    
    if len(labels) != len(args.csv_files):
        raise ValueError("Number of labels must match number of CSV files")
    
    compare_experiments(args.csv_files, labels, args.output_dir)


if __name__ == "__main__":
    main()

