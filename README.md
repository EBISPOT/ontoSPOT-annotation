# ontoSPOT-annotation
A set of tools that annotate free text to ontologies using LLMs.

# CurateGPT Batch Annotation

This project uses [CurateGPT](https://github.com/monarch-initiative/curategpt) for batch annotating ontology terms. CurateGPT is a prototype web application and framework for performing general purpose AI-guided curation and curation-related operations over collections of objects developed by the [Monarch Initiative](https://monarchinitiative.org/). 

## Setup Instructions

### Prerequisites

Before running the script, ensure you have the following:

1. **OpenAI API Key**: You need to set the `OPENAI_API_KEY` environment variable to authenticate with OpenAI.
   
   ```bash
   export OPENAI_API_KEY=<your_OpenAI_API_key>
   ```

### Setting Up the Environment

1. Clone this repository to your local machine:

	```bash
	git clone https://github.com/EBISPOT/ontoSPOT-annotation
	```

2. Create and set up the virtual environment:

	```bash
	make setup
	```

3. Index your ontologyusing `ont-<ontology_name>`, e.g.:

	```bash
	make ont-cl
	```

This will create a virtual environment and install the required dependencies.

## Usage

You can run batch annotation with the following command:

```bash
make annotate_batch ONT=cl IN=input_data/your_file.csv
```

Where:

- `ONT` is the ontology you want to use (e.g., `cl`, `uberon`, `efo`).
- `IN` is the input CSV file containing the terms to annotate (e.g., `input_data/terms.csv`).

The results will be saved in the `output_data` directory.

