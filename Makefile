# ==============================================================================
# RetailPulse Automation Makefile
# ==============================================================================

.PHONY: help install data clean features train test app api docker up down lint

PYTHON ?= .venv/Scripts/python.exe
PIP ?= .venv/Scripts/pip.exe

help:
	@echo "RetailPulse CLI Build Commands:"
	@echo "  make install    - Install required dependencies"
	@echo "  make data       - Download raw UCI/Kaggle dataset"
	@echo "  make clean      - Run data cleaning pipeline"
	@echo "  make features   - Extract RFM, time, lag, and churn features"
	@echo "  make train      - Train segmentation, forecasting, and churn models"
	@echo "  make test       - Run unit tests with pytest"
	@echo "  make app        - Launch Streamlit dashboard"
	@echo "  make api        - Launch FastAPI backend server"
	@echo "  make docker     - Build production Docker image"
	@echo "  make up         - Start full docker-compose stack"
	@echo "  make down       - Tear down docker-compose containers"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install prophet fastapi uvicorn httpx pyyaml optuna shap pytest pytest-cov

data:
	$(PYTHON) scripts/download_data.py

clean:
	$(PYTHON) scripts/run_clean.py

features:
	$(PYTHON) scripts/run_features.py

train:
	$(PYTHON) scripts/run_segmentation.py
	$(PYTHON) scripts/run_forecasting.py
	$(PYTHON) scripts/run_churn.py
	$(PYTHON) scripts/run_inventory.py
	$(PYTHON) scripts/run_monitoring.py

test:
	$(PYTHON) -m pytest tests/ -v

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port=8501

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker build -t retailpulse:latest .

up:
	docker-compose up -d

down:
	docker-compose down

lint:
	$(PYTHON) -m ruff check src/ tests/
