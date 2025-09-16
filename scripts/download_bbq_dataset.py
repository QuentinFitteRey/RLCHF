# scripts/download_bbq_dataset.py
from datasets import load_dataset
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, will use environment variables directly
    pass

DATA_DIR = os.getenv("DATA_DIR", "data")  # Default to "data" if not set


def download_and_save_bbq(local_data_dir: str = DATA_DIR):
    """Download BBQ dataset and save to local directory."""
    # create DATA_DIR/raw/bbq if it doesn't exist
    bbq_path = os.path.join(local_data_dir, "raw", "bbq")
    os.makedirs(bbq_path, exist_ok=True)
    
    print(f"Downloading BBQ dataset to {bbq_path}")
    load_dataset("heegyu/bbq", cache_dir=bbq_path)
    print("Download completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download BBQ bias benchmark dataset")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, 
                       help=f"Directory to save dataset (default: {DATA_DIR})")
    args = parser.parse_args()
    
    download_and_save_bbq(args.data_dir)
