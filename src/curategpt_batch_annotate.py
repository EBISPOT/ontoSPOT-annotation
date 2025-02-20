import csv
import subprocess
import os

# ------------------------- Configuration -------------------------
# Set the ontology name here (e.g., 'cl', 'uberon', 'efo')
ONTOLOGY = "cl"
# Path to the input CSV file (must contain a column named 'term')
INPUT_FILE = "../input_data/annotate_terms.csv"
# Path to the output CSV file
OUTPUT_FILE = os.path.join("../output_data", os.path.basename(os.path.splitext(INPUT_FILE)[0]) + "_results.csv")
# -----------------------------------------------------------------

def check_api_key():
    """Ensure the OPENAI_API_KEY environment variable is set."""
    if 'OPENAI_API_KEY' not in os.environ:
        raise EnvironmentError("Error: The OPENAI_API_KEY environment variable is not set. Use 'export OPENAI_API_KEY=<your_key>' to set it.")

def search_term(ontology: str, term: str):
    """Search for a term using curategpt and return results."""
    try:
        result = subprocess.run(
            ["curategpt", "search", "-c", f"terms_{ontology}", term],
            check=True, capture_output=True, text=True
        )
        return parse_search_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error searching for term '{term}': {e.stderr}")
        return []

def parse_search_output(output: str):
    """Parse curategpt search output and extract label, original_id, and distance."""
    candidates = []
    current_candidate = {}

    for line in output.strip().split("\n"):
        line = line.strip()
        if line.startswith("##") and "DISTANCE" in line:
            if current_candidate:
                candidates.append(current_candidate)
            distance = line.split("DISTANCE:")[-1].strip()
            current_candidate = {"distance": distance}
        elif line.startswith("label:"):
            current_candidate["label"] = line.split("label:", 1)[-1].strip()
        elif line.startswith("original_id:"):
            current_candidate["original_id"] = line.split("original_id:", 1)[-1].strip()

    if current_candidate:
        candidates.append(current_candidate)

    return candidates

def process_terms(input_file: str, output_file: str, ontology: str):
    """Read input terms, search candidates, and write results to output CSV."""
    with open(input_file, newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['term', 'candidate_label', 'candidate_original_id', 'candidate_distance']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            term = row.get('term', '').strip()
            if not term:
                print("Warning: Empty term encountered. Skipping.")
                continue

            print(f"Searching candidates for term: '{term}'...")
            candidates = search_term(ontology, term)
            if not candidates:
                writer.writerow({'term': term, 'candidate_label': '', 'candidate_original_id': '', 'candidate_distance': ''})
            else:
                for candidate in candidates:
                    writer.writerow({
                        'term': term,
                        'candidate_label': candidate.get('label', ''),
                        'candidate_original_id': candidate.get('original_id', ''),
                        'candidate_distance': candidate.get('distance', '')
                    })
            print(f"Finished processing term: '{term}'.")

def main():
    check_api_key()
    process_terms(INPUT_FILE, OUTPUT_FILE, ONTOLOGY)
    print(f"Results saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
