"""Data cleaning and preprocessing pipeline."""

import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
    
    return text.strip()


def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text and optionally remove stopwords."""
    if not text:
        return []
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Remove very short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens


def preprocess_dataset(df: pd.DataFrame, remove_stopwords: bool = True) -> pd.DataFrame:
    """Preprocess the entire dataset."""
    print("Cleaning text...")
    df['text_clean'] = df['statement'].apply(clean_text)
    
    print("Tokenizing text...")
    tqdm.pandas(desc="Tokenizing")
    df['tokens'] = df['text_clean'].progress_apply(
        lambda x: tokenize_text(x, remove_stopwords=remove_stopwords)
    )
    
    # Join tokens back to text for TF-IDF
    df['text_processed'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    # Remove empty texts
    df = df[df['text_processed'].str.len() > 0]
    
    print(f"After preprocessing: {len(df)} samples")
    
    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/validation/test sets."""
    # Use original splits if available
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'valid'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        print(f"Using original splits:")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
    else:
        # Random split if no original splits
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df['label_binary']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_binary']
        )
        
        print(f"Created random splits:")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def main() -> None:
    """Main function to clean and preprocess data."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "interim" / "liar_processed.csv"
    output_dir = project_root / "data" / "processed"
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        print("Please run download_liar.py first")
        sys.exit(1)
    
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples")
    
    # Preprocess
    df = preprocess_dataset(df, remove_stopwords=True)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Save processed datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    print(f"\nSaved processed datasets to: {output_dir}")
    print("\nFinal dataset statistics:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Show label distribution
    print("\nLabel distribution (0=Real, 1=Fake):")
    print("Train:", train_df['label_binary'].value_counts().sort_index().to_dict())
    print("Validation:", val_df['label_binary'].value_counts().sort_index().to_dict())
    print("Test:", test_df['label_binary'].value_counts().sort_index().to_dict())
    
    # Show sample
    print("\nSample processed text:")
    sample = train_df.iloc[0]
    print(f"Original: {sample['statement'][:100]}...")
    print(f"Processed: {sample['text_processed'][:100]}...")


if __name__ == "__main__":
    main()
