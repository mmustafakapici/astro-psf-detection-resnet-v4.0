# Default targets
.PHONY: all setup create-dirs download-test-data download-sdss-data annotate draw train test test-real clean help line-generate details
# Python environment
PYTHON=python

# Configuration file
CONFIG=config/config.yaml

# Default target
all: setup create-dirs download-test-data download-sdss-data line-generate annotate details draw control train 

# Kurulum
setup:
	@echo "Setting up environment..."
	pip install -r requirements.txt

create-dirs:
	@echo "Creating directories..."
	$(PYTHON) src/create_dirs.py

# Veri indirme ve çıkarma
download-test-data:
	@echo "Downloading and extracting data..."
	$(PYTHON) src/download_test_data.py

download-sdss-catalog:
	@echo "Downloading and extracting data..."
	$(PYTHON) src/download_sdss_fits_with_catalog.py

download-sdss-data:
	@echo "Downloading and extracting data..."
	$(PYTHON) src/download_sdss_data.py

# Annotasyon
annotate:
	@echo "Annotating code with type hints..."
	$(PYTHON) src/annotate.py

line-generate:
	@echo "Generating lines..."
	$(PYTHON) src/line_generator.py

details:
	@echo "Details..."
	$(PYTHON) src/extract_details.py
	
# Annotasyon çizme
draw:
	@echo "Drawing annotations on images sets..."
	$(PYTHON) src/draw_annotations.py


control:
	@echo "Controling annotations..."
	$(PYTHON) src/data_preprocessing.py

train:
	@echo "Training model..."
	$(PYTHON) src/train.py

train-DDP:
	@echo "Training model Distributing Data Parallel..."
	$(PYTHON) src/train-DDP.py

test:
	@echo "test data tests..."
	$(PYTHON) src/test_test_data.py

#modeli kullanarak real test
test-real:
	@echo "Real data..."
	$(PYTHON) src/test_real_data.py


# Temizlik
clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ */__pycache__
	rm -rf results/checkpoints/*
	rm -rf results/figures/*
	rm -rf data/processed/*
	rm -rf dist/*
	rm -rf build/*
	rm -rf *.egg-info


# Yardım
help:
	@echo "Available targets:"
	@echo "  make all:            Run all targets"
	@echo "  make create-dirs:    Create directories"
	@echo "  make download-test-data:  Download and extract tests data"
	@echo "  make download-sdss-data:  Download and extract sdss (train - val) data"
	@echo "  make annotate:       Annotate code with type hints"
	@echo "  make line-generate:  Generate lines"
	@echo "  make control:        Control annotations"
	@echo "  make details:        Extract details"
	@echo "  make draw:           Draw annotations on images sets"
	@echo "  make train:          Train model"
	@echo "  make test:           Run tests on test data"
	@echo "  make test-real:      Run tests on real data"
	@echo "  make clean:          Clean temporary files"
	@echo "  make setup:          Set up environment"
	@echo "  make help:           Show this help message"



	
