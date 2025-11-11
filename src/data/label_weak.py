"""Weak labeling system for multi-niche data."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class WeakLabeler:
    """Weak labeling system using heuristics and domain rules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.niche = config["niche"]
        self.positive_label = config["positive_label"]
        self.negative_label = config["negative_label"]

        # Compile regex patterns for better performance
        self.positive_phrases = [
            phrase.lower() for phrase in config["positive_phrases"]
        ]
        self.negative_phrases = [
            phrase.lower() for phrase in config["negative_phrases"]
        ]

        self.positive_domains = [
            domain.lower() for domain in config["positive_domains"]
        ]
        self.negative_domains = [
            domain.lower() for domain in config["negative_domains"]
        ]

        # Statistics tracking
        self.stats = {
            "total_samples": 0,
            "positive_labeled": 0,
            "negative_labeled": 0,
            "conflicts": 0,
            "unlabeled": 0,
            "domain_matches": 0,
            "phrase_matches": 0,
        }

    def label_text(
        self, text: str, source: str, domain_type: str
    ) -> Tuple[str, List[str]]:
        """Label a single text using weak labeling rules."""
        text_lower = text.lower()
        source_lower = source.lower()

        heuristics = []
        positive_score = 0
        negative_score = 0

        # Domain-based labeling
        if any(domain in source_lower for domain in self.positive_domains):
            positive_score += 2
            heuristics.append(f"positive_domain:{source}")

        if any(domain in source_lower for domain in self.negative_domains):
            negative_score += 2
            heuristics.append(f"negative_domain:{source}")

        # Phrase-based labeling
        positive_phrase_matches = []
        negative_phrase_matches = []

        for phrase in self.positive_phrases:
            if phrase in text_lower:
                positive_phrase_matches.append(phrase)
                positive_score += 1

        for phrase in self.negative_phrases:
            if phrase in text_lower:
                negative_phrase_matches.append(phrase)
                negative_score += 1

        if positive_phrase_matches:
            heuristics.append(f"positive_phrases:{','.join(positive_phrase_matches)}")

        if negative_phrase_matches:
            heuristics.append(f"negative_phrases:{','.join(negative_phrase_matches)}")

        # Determine label based on scores
        if positive_score > negative_score:
            label = self.positive_label
        elif negative_score > positive_score:
            label = self.negative_label
        else:
            # Tie or no matches - use domain type as fallback
            if domain_type == "positive":
                label = self.positive_label
                heuristics.append("domain_type_fallback:positive")
            elif domain_type == "negative":
                label = self.negative_label
                heuristics.append("domain_type_fallback:negative")
            else:
                label = "unlabeled"
                heuristics.append("no_heuristics_match")

        return label, heuristics

    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label entire dataset using weak labeling rules."""
        logger.info(f"Labeling {len(df)} samples for niche: {self.niche}")

        labeled_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling samples"):
            text = row["text"]
            source = row["source"]
            domain_type = row["domain_type"]

            label, heuristics = self.label_text(text, source, domain_type)

            labeled_data.append(
                {
                    "id": f"{self.niche}_{idx}",
                    "text": text,
                    "label": label,
                    "source": source,
                    "url": row["url"],
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "niche": self.niche,
                    "heuristics": "|".join(heuristics),
                    "title": row.get("title", ""),
                    "domain_type": domain_type,
                }
            )

        labeled_df = pd.DataFrame(labeled_data)

        # Update statistics
        self._update_stats(labeled_df)

        return labeled_df

    def _update_stats(self, df: pd.DataFrame) -> None:
        """Update labeling statistics."""
        self.stats["total_samples"] = len(df)
        self.stats["positive_labeled"] = len(df[df["label"] == self.positive_label])
        self.stats["negative_labeled"] = len(df[df["label"] == self.negative_label])
        self.stats["unlabeled"] = len(df[df["label"] == "unlabeled"])

        # Count conflicts (samples with both positive and negative heuristics)
        conflicts = 0
        domain_matches = 0
        phrase_matches = 0

        for heuristics in df["heuristics"]:
            if "positive_domain" in heuristics and "negative_domain" in heuristics:
                conflicts += 1
            if "positive_domain" in heuristics or "negative_domain" in heuristics:
                domain_matches += 1
            if "positive_phrases" in heuristics or "negative_phrases" in heuristics:
                phrase_matches += 1

        self.stats["conflicts"] = conflicts
        self.stats["domain_matches"] = domain_matches
        self.stats["phrase_matches"] = phrase_matches

    def generate_quality_report(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Generate label quality report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        report_content = f"""# Label Quality Report: {self.niche}

## Overview
- **Niche**: {self.niche}
- **Total Samples**: {self.stats['total_samples']}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Label Distribution
- **{self.positive_label}**: {self.stats['positive_labeled']} ({self.stats['positive_labeled']/self.stats['total_samples']*100:.1f}%)
- **{self.negative_label}**: {self.stats['negative_labeled']} ({self.stats['negative_labeled']/self.stats['total_samples']*100:.1f}%)
- **Unlabeled**: {self.stats['unlabeled']} ({self.stats['unlabeled']/self.stats['total_samples']*100:.1f}%)

## Heuristic Coverage
- **Domain Matches**: {self.stats['domain_matches']} ({self.stats['domain_matches']/self.stats['total_samples']*100:.1f}%)
- **Phrase Matches**: {self.stats['phrase_matches']} ({self.stats['phrase_matches']/self.stats['total_samples']*100:.1f}%)
- **Conflicts**: {self.stats['conflicts']} ({self.stats['conflicts']/self.stats['total_samples']*100:.1f}%)

## Configuration Used
- **Positive Phrases**: {', '.join(self.config['positive_phrases'][:5])}...
- **Negative Phrases**: {', '.join(self.config['negative_phrases'][:5])}...
- **Positive Domains**: {', '.join(self.config['positive_domains'][:5])}...
- **Negative Domains**: {', '.join(self.config['negative_domains'][:5])}...

## Sample Data (First 10 Rows)
"""

        # Add sample data
        sample_df = df.head(10)[["id", "label", "source", "heuristics"]].copy()
        sample_df["text_preview"] = df.head(10)["text"].str[:100] + "..."

        report_content += sample_df.to_string(index=False)

        report_content += """

## Manual Audit Checklist
- [ ] Review 100 random samples for label accuracy
- [ ] Check for systematic biases in domain-based labeling
- [ ] Verify phrase-based heuristics are appropriate
- [ ] Assess class balance and potential improvements
- [ ] Consider additional heuristics for unlabeled samples

## Limitations & Biases
- **Domain Bias**: Labels heavily influenced by source domain
- **Phrase Bias**: May miss nuanced content that doesn't contain key phrases
- **Temporal Bias**: Heuristics may not adapt to changing language patterns
- **Cultural Bias**: English-only phrases may miss non-English content
- **Context Bias**: Short texts may lack sufficient context for accurate labeling

## Recommendations
1. **Manual Review**: Audit at least 100 samples for each class
2. **Iterative Improvement**: Refine heuristics based on manual review
3. **Balance Check**: Ensure reasonable class distribution
4. **Quality Threshold**: Consider minimum sample requirements per class
5. **Validation Split**: Set aside data for manual validation
"""

        # Save report
        report_path = output_dir / "label_quality.md"
        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"Saved label quality report to {report_path}")


def load_niche_config(niche: str) -> Dict[str, Any]:
    """Load configuration for a specific niche."""
    config_path = Path(__file__).parent.parent.parent / "configs" / f"{niche}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Main function for weak labeling."""
    parser = argparse.ArgumentParser(description="Apply weak labeling to niche data")
    parser.add_argument("--niche", required=True, help="Niche to label data for")

    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / args.niche / "raw" / "scraped_data.csv"
    output_dir = project_root / "data" / args.niche / "processed"
    reports_dir = project_root / "artifacts" / args.niche / "reports"

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run scraping pipeline first")
        sys.exit(1)

    try:
        # Load config and data
        config = load_niche_config(args.niche)
        df = pd.read_csv(input_file)

        logger.info(f"Loaded {len(df)} samples for labeling")

        # Create labeler and label data
        labeler = WeakLabeler(config)
        labeled_df = labeler.label_dataset(df)

        # Check minimum samples per class
        min_samples = config.get("min_samples_per_class", 300)
        positive_count = len(
            labeled_df[labeled_df["label"] == config["positive_label"]]
        )
        negative_count = len(
            labeled_df[labeled_df["label"] == config["negative_label"]]
        )

        if positive_count < min_samples or negative_count < min_samples:
            logger.warning("Insufficient samples for training:")
            logger.warning(
                f"  {config['positive_label']}: {positive_count} (minimum: {min_samples})"
            )
            logger.warning(
                f"  {config['negative_label']}: {negative_count} (minimum: {min_samples})"
            )
            logger.warning("Consider collecting more data or adjusting heuristics")

        # Save labeled data
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "train.csv"
        labeled_df.to_csv(output_file, index=False)

        logger.info(f"Saved {len(labeled_df)} labeled samples to {output_file}")

        # Generate quality report
        labeler.generate_quality_report(labeled_df, reports_dir)

        # Print summary
        logger.info(f"\nLabeling Summary for {args.niche}:")
        logger.info(f"Total samples: {len(labeled_df)}")
        logger.info(f"Positive ({config['positive_label']}): {positive_count}")
        logger.info(f"Negative ({config['negative_label']}): {negative_count}")
        logger.info(f"Unlabeled: {len(labeled_df[labeled_df['label'] == 'unlabeled'])}")
        logger.info(f"Domain matches: {labeler.stats['domain_matches']}")
        logger.info(f"Phrase matches: {labeler.stats['phrase_matches']}")
        logger.info(f"Conflicts: {labeler.stats['conflicts']}")

    except Exception as e:
        logger.error(f"Error labeling data for {args.niche}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
