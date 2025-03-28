import os
import pandas as pd
from openai import OpenAI
import sys

# Set up the OpenAI client using the API key from environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to query ChatGPT and get potential HP terms for an EFO term
def get_matching_hp_terms(efo_term):
    instructions = f"Given the EFO term '{efo_term}', which human phenotype ontology (HP) terms could be associated with it? Provide only the HP term labels in a comma-separated list, without IDs or additional text."
    
    try:
        # Requesting the response from the GPT-4o model
        response = client.responses.create(
            model="gpt-4o",  # Use the appropriate model name
            instructions=instructions,
            input=efo_term,
        )
        
        # Extract and clean the HP terms from the response
        hp_terms = response.output_text.strip()
        
        # Clean the response text by splitting on commas and stripping unwanted text
        cleaned_terms = [term.strip() for term in hp_terms.split(',') if len(term.strip()) > 0]
        
        return cleaned_terms
    
    except Exception as e:
        print(f"Error with EFO term '{efo_term}': {e}")
        return None

# Function to process EFO terms and retrieve matching HP terms
def process_efo_terms(efo_terms):
    data = []
    
    for efo_term in efo_terms:
        print(f"Processing EFO term: {efo_term}")
        hp_terms = get_matching_hp_terms(efo_term)
        
        if hp_terms:
            # For each HP term, create a new row with the EFO term and HP term
            for hp_term in hp_terms:
                data.append({'EFO Term': efo_term, 'HP Term': hp_term})
        else:
            # In case there are no matching HP terms, append a single row with no matches
            data.append({'EFO Term': efo_term, 'HP Term': "No matches found"})
    
    # Convert to DataFrame for easy manipulation and export
    df = pd.DataFrame(data)
    return df

# Main function that accepts the input CSV file from the command line argument
def main(input_csv):
    # Load your EFO terms from the input CSV
    efo_terms = pd.read_csv(input_csv)['EFO Term'].tolist()  # Adjust this if your file structure differs

    # Process the EFO terms and find matching HP terms
    hp_mappings = process_efo_terms(efo_terms)

    # Ensure output_data directory exists
    output_dir = "output_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the output file path within the 'output_data' directory
    output_file = os.path.join(output_dir, os.path.basename(input_csv).replace('.csv', '_to_hp_mappings.csv'))

    # Save the results to a CSV file
    hp_mappings.to_csv(output_file, index=False)

    print(f"Mapping completed and saved to '{output_file}'")

if __name__ == "__main__":
    input_csv = sys.argv[1]  # Get the input file from the command line
    main(input_csv)
