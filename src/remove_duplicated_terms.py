import pandas as pd
import argparse
import os

def remove_duplicates_from_csv(file_path):
    """Removes duplicate rows from a specified CSV file and overwrites it."""
    try:
        df = pd.read_csv(file_path)
        df.drop_duplicates(inplace=True)
        df.to_csv(file_path, index=False)
        print(f"Processed: {file_path}, removed duplicates.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate terms from a CSV file.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: File {args.input} does not exist.")
    else:
        remove_duplicates_from_csv(args.input)