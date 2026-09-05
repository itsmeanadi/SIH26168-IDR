.PHONY: help setup data train eval plots export demo test clean

PYTHON ?= python

help:
	@echo "AI-ML Intelligent Dead Reckoning (IDR) System - SIH PS 26168 (ISRO)"
	@echo "Available commands:"
	@echo "  make setup   : Install dependencies in active Python environment"
	@echo "  make data    : Ingest/download IO-VNBD dataset, log schema & preprocess"
	@echo "  make train   : Train IMU denoiser, forward-velocity estimator, and residual nets"
	@echo "  make eval    : Run GNSS blackout simulation, DR + NHC + Map-matching"
	@echo "  make plots   : Generate trajectory, drift-vs-dist, speed & map overlay plots in reports/"
	@echo "  make export  : Run ONNX and TFLite model export smoke tests"
	@echo "  make demo    : Launch Streamlit trajectory playback demo (optional)"
	@echo "  make test    : Run pytest unit tests"
	@echo "  make clean   : Remove cached build, temp and bytecode files"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) scripts/download_data.py --output-dir data/raw
	$(PYTHON) scripts/prepare_data.py --raw-dir data/raw --output-dir data/processed

train:
	$(PYTHON) -m idr.models.train_all --data-dir data/processed --output-dir models/

eval:
	$(PYTHON) -m idr.eval.blackout --data-dir data/processed --model-dir models/ --report-dir reports/

plots:
	$(PYTHON) -m idr.eval.plotting --report-dir reports/ --output-dir reports/figures

export:
	$(PYTHON) -m idr.export.convert --model-dir models/ --output-dir models/export

demo:
	$(PYTHON) -m streamlit run src/idr/demo/app.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
