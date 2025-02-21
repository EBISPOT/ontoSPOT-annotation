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
	@echo "  clean            Remove output CSV files from output_data/"
	@echo "  setup            Set up the virtual environment and install dependencies"
	@echo "  help             Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make annotate_batch                   # Run with defaults"
	@echo "  make annotate_batch ONT=uberon IN=input_data/terms.csv"
	@echo "  make clean                             # Clean output files"
	@echo "  make setup                             # Create virtual environment and install requirements"
	@echo ""

# Run the batch annotation script
annotate_batch:
	python3 ./src/curategpt_batch_annotate.py --ontology $(ONT) --input $(IN)

# Clean output CSV files
clean:
	rm -f output_data/*_results.csv
	@echo "🧹 Output files cleaned."

# Set up virtual environment and install dependencies
setup:
	python3 -m venv curate_venv
	. curate_venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Virtual environment created and dependencies installed."

# ------------------------- End -------------------------
