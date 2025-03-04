import csv
import openai
import os
import argparse

# Check if the OpenAI API key is set
def check_api_key():
    if 'OPENAI_API_KEY' not in os.environ:
        raise EnvironmentError("Error: The OPENAI_API_KEY environment variable is not set. Use 'export OPENAI_API_KEY=<your_key>' to set it.")

# Function to query OpenAI's GPT model to get the expanded form of abbreviations
def expand_abbreviation_with_chatgpt(term: str, domain: str):
    """Query ChatGPT to expand an abbreviation into its full form."""
    try:
        # Make sure the model name is correct and the prompt is clear, including domain context
        response = openai.chat.completions.create(
            model="gpt-4o",  # You can use other models as needed (e.g., "gpt-3.5-turbo")
            messages=[
                {"role": "user", "content": f"In the context of {domain}, is '{term}' an abbreviation? If it is, provide the full expanded form. Also, if you encounter a '+' symbol, it is an abbreviation for '-positive'. The '+' also indicates that it is positive for a biomarker (e.g.. CD4+ is CD4-positive), and the biomarker should not be altered (e.g., CD4+ will become 'CD4-positive', CD4 remains as it is). In your answer, please only provide the full expanded form, not any other text. If it is not an abbreviation, return the exact same term."}
            ],
            max_tokens=50,
            temperature=0.5,
        )

        # Correct way to access the message content
        expanded_term = response.choices[0].message.content.strip()

        # If the expanded term is empty or the same as the original term, return the original term
        if not expanded_term or expanded_term.lower() == term.lower():
            return term
        return expanded_term

    except Exception as e:
        print(f"Error querying ChatGPT for term '{term}': {e}")
        return term  # Return the original term in case of an error

# Function to handle abbreviation checking for terms in a CSV file
def process_terms(input_file: str, domain: str):
    """Read input terms from a CSV, check abbreviations, and write expanded forms to output CSV."""
    # Generate output file name by appending '_non_abbreviated' to the input file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"output_data/{base_name}_non_abbreviated.csv"

    with open(input_file, newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['term', 'expanded_term']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            term = row.get('term', '').strip()
            if not term:
                print("Warning: Empty term encountered. Skipping.")
                continue

            print(f"Checking term: '{term}'...")
            expanded_term = expand_abbreviation_with_chatgpt(term, domain)
            writer.writerow({'term': term, 'expanded_term': expanded_term})
            print(f"Term '{term}' expanded to '{expanded_term}'.")

    print(f"Results saved to '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(description="Batch expand abbreviations using ChatGPT.")
    parser.add_argument('--input', dest='input_file', required=True, help="Path to the input CSV file.")
    parser.add_argument('--domain', dest='domain', required=True, help="Domain for context (e.g., 'cell types').")
    args = parser.parse_args()

    check_api_key()  # Ensure the OpenAI API key is set
    process_terms(args.input_file, args.domain)

if __name__ == "__main__":
    main()