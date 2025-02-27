# Makefile for running curategpt_batch_annotate.py with specified variables

# ------------------------- Makefile -------------------------
# Abbreviations:
# ONT = ontology (e.g., cl, uberon, efo)
# IN = input file path (e.g., input_data/example.csv)

ONT ?= cl
IN ?= input_data/example.csv

# ------------------------- Targets -------------------------

# Display help message
help:
	@echo ""
	@echo "📝 Usage: make <target> [ONT=ontology] [IN=input_file]"
	@echo ""
	@echo "Targets:"
	@echo "  annotate_batch   Run batch annotation (default: ONT=$(ONT), IN=$(IN))"
	@echo "  onto-<ontology>  Download ontology and create index (e.g., onto-uberon)"
	@echo "  clean_output            Remove output CSV files from output_data/"
	@echo "  clean_db            Remove databases files from db/"
	@echo "  setup            Set up the virtual environment and install dependencies"
	@echo "  help             Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make annotate_batch                   # Run with defaults"
	@echo "  make onto-uberon                      # Download and index the Uberon ontology"
	@echo "  make annotate_batch ONT=uberon IN=input_data/terms.csv"
	@echo "  make clean_output                      # Clean output files"
	@echo "  make setup                             # Create virtual environment and install requirements"
	@echo ""

# Run the batch annotation script
annotate_batch:
	python3 ./src/curategpt_batch_annotate.py --ontology $(ONT) --input $(IN)

# Define a generic target for downloading the ontology and creating the index
onto-%:
	curategpt ontology index -m openai: -c terms_$* sqlite:obo:$*

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
	python3 -m venv curate_venv
	. curate_venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Virtual environment created and dependencies installed."

# ------------------------- End -------------------------
