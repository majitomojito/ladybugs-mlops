.PHONY: help install dev test train clean docker-build docker-run all

help:
	@echo "MLOps Demo Project - Available Commands"
	@echo "========================================"
	@echo "make install      - Install dependencies"
	@echo "make dev          - Install development dependencies"
	@echo "make test         - Run tests"
	@echo "make train        - Run the ML pipeline"
	@echo "make clean        - Remove generated artifacts"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run   - Run pipeline in Docker"
	@echo "make all          - Install, test, and train"

install:
	pip install -r requirements.txt

dev: install
	pip install pytest pytest-cov

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python train.py

clean:
	rm -rf data/*.npy models/*.pkl models/metrics/*.json __pycache__ .pytest_cache htmlcov .coverage

docker-build:
	docker build -t mlops-demo:latest .

docker-run: docker-build
	docker run -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data mlops-demo:latest

all: install test train
	@echo "MLOps pipeline complete!"
