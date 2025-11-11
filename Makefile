.PHONY: setup data train evaluate predict clean test lint format

# Environment setup
setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -e ".[dev]"
	. venv/bin/activate && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
	. venv/bin/activate && pre-commit install

# Original LIAR dataset pipeline (backwards compatibility)
data-liar:
	. venv/bin/activate && python -m src.data.download_liar
	. venv/bin/activate && python -m src.data.clean

train-liar:
	. venv/bin/activate && python -m src.models.train

evaluate-liar:
	. venv/bin/activate && python -m src.models.eval

# Multi-niche data pipeline
data:
	@if [ -z "$(NICHE)" ]; then \
		echo "Usage: make data NICHE=<niche_name>"; \
		echo "Available niches: climate_claims, celebrity_rumors, hustle_scams, ai_hype, campus_rumor, medical_claims, crypto_scams"; \
	else \
		. venv/bin/activate && python -m src.data.pipeline --niche $(NICHE); \
	fi

# Multi-niche training
train:
	@if [ -z "$(NICHE)" ]; then \
		echo "Usage: make train NICHE=<niche_name>"; \
		echo "Available niches: climate_claims, celebrity_rumors, hustle_scams, ai_hype, campus_rumor, medical_claims, crypto_scams"; \
	else \
		. venv/bin/activate && python -m src.models.train_niche --niche $(NICHE); \
	fi

# Multi-niche evaluation
evaluate:
	@if [ -z "$(NICHE)" ]; then \
		echo "Usage: make evaluate NICHE=<niche_name>"; \
		echo "Available niches: climate_claims, celebrity_rumors, hustle_scams, ai_hype, campus_rumor, medical_claims, crypto_scams"; \
	else \
		. venv/bin/activate && python -m src.models.eval_niche --niche $(NICHE); \
	fi

# Unified multi-niche training
unified-train:
	. venv/bin/activate && python -m src.models.train_unified --niches climate_claims,celebrity_rumors,hustle_scams,ai_hype,campus_rumor,medical_claims,crypto_scams

unified-eval:
	. venv/bin/activate && python -m src.models.eval_unified --report_by_niche true

# Prediction
predict:
	@if [ -z "$(TEXT)" ]; then \
		echo "Usage: make predict TEXT=\"your text here\""; \
	else \
		. venv/bin/activate && python -m src.predict.cli --text "$(TEXT)"; \
	fi

# Batch prediction
predict-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make predict-file FILE=path/to/file.csv"; \
	else \
		. venv/bin/activate && python -m src.predict.cli --file "$(FILE)"; \
	fi

# Run Streamlit demo
demo:
	. venv/bin/activate && streamlit run app/streamlit_app.py

# Testing
test:
	. venv/bin/activate && python -m pytest tests/ -v

test-cov:
	. venv/bin/activate && python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-security:
	. venv/bin/activate && python -m pytest tests/test_security.py -v

# Linting and formatting
lint:
	. venv/bin/activate && ruff check src/ tests/
	. venv/bin/activate && flake8 src/ tests/ --count --statistics
	. venv/bin/activate && mypy src/ --ignore-missing-imports || true

lint-all:
	. venv/bin/activate && ruff check src/ tests/
	. venv/bin/activate && flake8 src/ tests/ --count --statistics
	. venv/bin/activate && pylint src/ --rcfile=.pylintrc || true
	. venv/bin/activate && mypy src/ --ignore-missing-imports || true

format:
	. venv/bin/activate && black src/ tests/
	. venv/bin/activate && isort src/ tests/
	. venv/bin/activate && ruff format src/ tests/

# Security checks
security:
	. venv/bin/activate && pip install bandit safety || true
	. venv/bin/activate && bandit -r src -ll || true
	. venv/bin/activate && safety check || true
	. venv/bin/activate && python -m pytest tests/test_security.py -v --no-cov

# Clean artifacts
clean:
	rm -rf artifacts/models/
	rm -rf artifacts/vectorizers/
	rm -rf artifacts/reports/
	rm -rf data/processed/
	rm -rf data/interim/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Full pipeline run
full-pipeline: setup data train evaluate

# Help
help:
	@echo "Available commands:"
	@echo "  setup          - Set up virtual environment and install dependencies"
	@echo ""
	@echo "Multi-niche commands:"
	@echo "  data NICHE=<name>     - Scrape and label data for specific niche"
	@echo "  train NICHE=<name>    - Train model for specific niche"
	@echo "  evaluate NICHE=<name> - Evaluate model for specific niche"
	@echo "  unified-train         - Train unified model across all niches"
	@echo "  unified-eval          - Evaluate unified model"
	@echo ""
	@echo "Available niches: climate_claims, celebrity_rumors, hustle_scams,"
	@echo "                  ai_hype, campus_rumor, medical_claims, crypto_scams"
	@echo ""
	@echo "Legacy LIAR dataset commands:"
	@echo "  data-liar      - Download and preprocess LIAR dataset"
	@echo "  train-liar     - Train model on LIAR dataset"
	@echo "  evaluate-liar  - Evaluate LIAR model"
	@echo ""
	@echo "Other commands:"
	@echo "  predict        - Predict on a single text (usage: make predict TEXT=\"...\")"
	@echo "  predict-file   - Predict on a CSV file (usage: make predict-file FILE=...)"
	@echo "  demo           - Run Streamlit demo with niche selection"
	@echo "  test           - Run tests"
	@echo "  test-cov       - Run tests with coverage report"
	@echo "  test-security  - Run security tests"
	@echo "  lint           - Run linting checks (ruff, flake8, mypy)"
	@echo "  lint-all       - Run all linting checks (ruff, flake8, pylint, mypy)"
	@echo "  format         - Format code (black, isort, ruff)"
	@echo "  security       - Run security audit (bandit, safety, security tests)"
	@echo "  clean          - Clean generated artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make data NICHE=climate_claims"
	@echo "  make train NICHE=climate_claims"
	@echo "  make unified-train"
