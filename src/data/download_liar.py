"""Download and prepare the LIAR dataset."""

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def download_file(url: str, filepath: Path) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    total_size = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        desc=f"Downloading {filepath.name}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def download_liar_dataset(data_dir: Path) -> None:
    """Download the LIAR dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # LIAR dataset URLs
    urls = {
        "train": "https://raw.githubusercontent.com/thiagorainmaker/liar-liar-pants-on-fire/master/liar_dataset/train.tsv",
        "test": "https://raw.githubusercontent.com/thiagorainmaker/liar-liar-pants-on-fire/master/liar_dataset/test.tsv",
        "valid": "https://raw.githubusercontent.com/thiagorainmaker/liar-liar-pants-on-fire/master/liar_dataset/valid.tsv",
    }

    for split, url in urls.items():
        filepath = data_dir / f"{split}.tsv"
        if not filepath.exists():
            print(f"Downloading {split} set...")
            download_file(url, filepath)
        else:
            print(f"{split} set already exists: {filepath}")


def parse_liar_dataset(data_dir: Path) -> pd.DataFrame:
    """Parse LIAR dataset and combine splits."""
    columns = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state_info",
        "party_affiliation",
        "barely_true_count",
        "false_count",
        "half_true_count",
        "mostly_true_count",
        "pants_on_fire_count",
        "context",
    ]

    all_data = []

    for split in ["train", "valid", "test"]:
        filepath = data_dir / f"{split}.tsv"
        if filepath.exists():
            df = pd.read_csv(filepath, sep="\t", header=None, names=columns)
            df["split"] = split
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {split} set")

    if not all_data:
        raise FileNotFoundError(
            "No LIAR dataset files found. Run download_liar_dataset first."
        )

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples")

    return combined_df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map LIAR labels to binary classification."""
    # LIAR has 6 classes, we'll map to binary: True/Real vs False/Fake
    label_mapping = {
        "true": 0,  # Real news
        "mostly-true": 0,  # Real news
        "half-true": 0,  # Real news (could be debated, but not intentionally fake)
        "barely-true": 1,  # Fake news
        "false": 1,  # Fake news
        "pants-fire": 1,  # Fake news
    }

    df["label_binary"] = df["label"].map(label_mapping)

    # Remove rows with unmapped labels
    df = df.dropna(subset=["label_binary"])
    df["label_binary"] = df["label_binary"].astype(int)

    print("Label distribution:")
    print(df["label_binary"].value_counts().sort_index())

    return df


def main() -> None:
    """Main function to download and process LIAR dataset."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"

    print("Downloading LIAR dataset...")
    download_liar_dataset(data_dir)

    print("Parsing LIAR dataset...")
    df = parse_liar_dataset(data_dir)

    print("Mapping labels to binary classification...")
    df = map_labels(df)

    # Save processed data
    output_dir = project_root / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "liar_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved processed dataset to: {output_file}")

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print("Label distribution (0=Real, 1=Fake):")
    print(df["label_binary"].value_counts().sort_index())

    # Show sample
    print("\nSample data:")
    print(df[["statement", "label", "label_binary"]].head())


if __name__ == "__main__":
    main()
