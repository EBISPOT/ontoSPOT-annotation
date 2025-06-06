import csv
import subprocess
import os
import argparse
from oaklib import get_adapter

# ------------------------- Configuration -------------------------
# Variables are passed via command-line arguments (ONTOLOGY and INPUT_FILE)
# OUTPUT_FILE is automatically derived from INPUT_FILE.
# -----------------------------------------------------------------

def check_api_key():
    """Ensure the OPENAI_API_KEY environment variable is set."""
    if 'OPENAI_API_KEY' not in os.environ:
        raise EnvironmentError("Error: The OPENAI_API_KEY environment variable is not set. Use 'export OPENAI_API_KEY=<your_key>' to set it.")

def search_term(ontology: str, term: str):
    """Search for a term using curategpt and return results."""
    try:
        result = subprocess.run(
            ["curategpt", "search", "-c", f"terms_{ontology}", term, "-l", "5"],
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

    oak_onto = get_adapter(f'sqlite:obo:{ontology}')
    exact_matches = {}
    for curie in oak_onto.entities():
        label = oak_onto.label(curie)
        exact_matches[label] = {'curie': curie, 'label': label}
        alias_map = oak_onto.entity_alias_map(curie)
        if 'oio:hasExactSynonym' in alias_map:
            for alias in alias_map['oio:hasExactSynonym']:
                exact_matches[alias] = {'curie': curie, 'label': alias}

    """Read input terms, search candidates, and write results to output CSV."""
    with open(input_file, newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames+['candidate_label', 'candidate_original_id', 'candidate_distance']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            term = row.get('term', '').strip()
            if not term:
                print("Warning: Empty term encountered. Skipping.")
                continue


            print(f"Searching for exact lexical matches for term: '{term}'...")
            if term in exact_matches:
                exact_match = exact_matches[term]
                res_row = {}
                res_row.update(row)
                res_row.update({
                    'candidate_label': exact_match['label'],
                    'candidate_original_id': exact_match['curie'],
                    'candidate_distance': 0
                })
                writer.writerow(res_row)
                print(f"    - exact match found, will not search embeddings")
                continue

            print(f"Searching embeddings for term: '{term}'...")


            candidates = search_term(ontology, term)
            if not candidates:
                writer.writerow({'term': term, 'candidate_label': '', 'candidate_original_id': '', 'candidate_distance': ''})
            else:
                subsumed = False
                for candidate in candidates:
                    res_row = {}
                    res_row.update(row)
                    res_row.update({
                        'candidate_label': candidate.get('label', ''),
                        'candidate_original_id': candidate.get('original_id', ''),
                        'candidate_distance': candidate.get('distance', '')
                    })
                    writer.writerow(res_row)
            print(f"Finished processing term: '{term}'.")

def main():
    parser = argparse.ArgumentParser(description="Batch annotate terms using curateGPT.")
    parser.add_argument('--ontology', required=True, help="Ontology to use (e.g., 'cl', 'uberon', 'efo').")
    parser.add_argument('--input', dest='input_file', required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    ontology = args.ontology
    input_file = args.input_file
    output_file = os.path.join("./output_data", os.path.basename(os.path.splitext(input_file)[0]) + "_results.csv")

    check_api_key()
    process_terms(input_file, output_file, ontology)
    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()