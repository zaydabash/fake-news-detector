"""Multi-niche data pipeline orchestrator."""

import sys
from pathlib import Path
import argparse

from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_data_pipeline(niche: str) -> None:
    """Run complete data pipeline for a niche."""
    logger.info(f"Starting data pipeline for niche: {niche}")
    
    # Import here to avoid circular imports
    from src.data.scrape_sources import main as scrape_main
    from src.data.label_weak import main as label_main
    
    # Set up argument parsing for subprocess calls
    import sys
    original_argv = sys.argv.copy()
    
    try:
        # Run scraping
        logger.info("Step 1: Scraping data sources...")
        sys.argv = ['scrape_sources.py', '--niche', niche]
        scrape_main()
        
        # Run weak labeling
        logger.info("Step 2: Applying weak labeling...")
        sys.argv = ['label_weak.py', '--niche', niche]
        label_main()
        
        logger.info(f"Data pipeline completed for niche: {niche}")
        
    except Exception as e:
        logger.error(f"Data pipeline failed for niche {niche}: {e}")
        raise
    finally:
        sys.argv = original_argv


def main():
    """Main function for data pipeline."""
    parser = argparse.ArgumentParser(description="Run data pipeline for a niche")
    parser.add_argument("--niche", required=True, help="Niche to process")
    
    args = parser.parse_args()
    
    try:
        run_data_pipeline(args.niche)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
