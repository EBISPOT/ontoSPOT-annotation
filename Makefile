# Makefile for running curategpt_batch_annotate.py with specified variables

# ------------------------- Makefile -------------------------
# Abbreviations:
# ONT = ontology (e.g., cl, uberon, efo)
# IN = input file path (e.g., input_data/example.csv)
# DOMAIN = Indicates the domain of the abbreviations (e.g., cell type, disease)

ONT ?= cl
IN ?= input_data/example.csv
DOMAIN ?= 'cell type'

# ------------------------- Targets -------------------------
all: help # Default target

# Display help message
help:
	@echo ""
	@echo "📝 Usage: make <target> [ONT=ontology] [IN=input_file]"
	@echo ""
	@echo "Targets:"
	@echo "  annotate_batch   Run batch annotation (default: ONT=cl, IN=input_data/example.csv))"
	@echo "  expand_abbreviations  Expand abbreviations in the input CSV file using ChatGPT (default: IN=input_data/example.csv)"
	@echo "  onto-<ontology>  Download ontology and create index (e.g., onto-uberon)"
	@echo "  remove_duplicates  Remove duplicated terms in a list (default: IN=input_data/terms.csv)"
	@echo "  clean_output            Remove output CSV files from output_data/"
	@echo "  clean_db            Remove databases files from db/"
	@echo "  setup            Set up the virtual environment and install dependencies"
	@echo "  help             Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make annotate_batch                   # Run with defaults"
	@echo "  make onto-uberon                      # Download and index the Uberon ontology"
	@echo "  make annotate_batch ONT=uberon IN=input_data/terms.csv"
	@echo "  make expand_abbreviations IN=input_data/terms.csv DOMAIN='anatomical parts'
	@echo "  make remove_duplicates IN=input_data/terms.csv"
	@echo ""

# Run the batch annotation script
annotate_batch:
	python3 ./src/curategpt_batch_annotate.py --ontology $(ONT) --input $(IN)

expand_abbreviations:
	python3 ./src/expand_abbreviations.py --input $(IN) --domain $(DOMAIN)

# Define a generic target for downloading the ontology and creating the index
onto-%:
	curategpt ontology index -m openai: --index-fields label,definition,relationships,aliases -c terms_$* sqlite:obo:$*

# Remove duplicated terms in a list
remove_duplicates:
	python3 src/remove_duplicated_terms.py --input $(IN)

# Clean output CSV files
clean_output:
	rm -f output_data/*_results.csv
	@echo "🧹 Output files cleaned."

# Clean databases
clean_db:
	rm -r db/*
	@echo "🧹 Databases cleaned."

# Set up virtual environment and install dependencies
setup:
	python3.12 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Virtual environment created and dependencies installed."

# ------------------------- End -------------------------
