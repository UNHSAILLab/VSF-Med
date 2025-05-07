# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VSF-Med is a Vulnerability Scoring Framework for Medical Vision-Language Models. It evaluates the safety, reliability, and adversarial robustness of Vision-Language Models in clinical imaging applications using an ordinal 0-4 scale across eight vulnerability dimensions.

## Repository Structure

The repository is organized around:
- `src/`: Source code for the framework components
- `notebooks/`: Jupyter notebooks for running experiments
- `templates/`: Text templates for experiments
- Database schema for storing results (PostgreSQL)

## Setup and Configuration

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Database Setup

1. Install PostgreSQL 13+
2. Create database schema:
```bash
psql -U username -d database_name -f src/database/dbschema.sql
```
3. Update database connection settings in your config file

## Common Commands

### Running Experiments

Generate adversarial prompts:
```bash
python src/data/processed/generate_adversarial_prompts.py --config src/config/my_config.yaml
```

Generate visual perturbations:
```bash
python src/utils/perturbations/image_perturbations.py --source /path/to/images --output /path/to/output
```

### Running Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook

# Run specific experiments
jupyter notebook notebooks/01_generate_adversarial_samples.ipynb
jupyter notebook notebooks/04_gpt_vulnerability_evaluation.ipynb
```

## Core Framework Components

### 1. Vulnerability Dimensions

VSF-Med evaluates models on eight vulnerability dimensions:
- Prompt injection effectiveness
- Jailbreak resilience
- Potential confidentiality breach
- Risk of misinformation
- Denial of service resilience
- Persistence of attack effects
- Safety bypass success
- Impact on medical decision support

### 2. Text Attack Categories

The framework implements 18 different text attack categories including:
- Prompt Injection
- Jailbreak Attempts
- Confidentiality Breach
- Misinformation Generation
- And more...

### 3. Visual Perturbation Methods

Six visual perturbation techniques:
- Gaussian noise at various levels
- Checkerboard overlays
- Moiré patterns
- Random arrow artifacts
- Steganographic information hiding
- LSB-plane extraction

### 4. Evaluation Pipeline

The evaluation workflow:
1. Dataset selection (MIMIC-CXR)
2. Adversarial variant generation
3. Model inference through API clients
4. LLM-based scoring using VSF rubric
5. Aggregation and analysis

## Libraries and Dependencies

Key libraries:
- Data processing: pandas, numpy
- Image processing: Pillow, opencv-python, scikit-image
- Database: sqlalchemy, psycopg2-binary
- API integration: openai, requests
- Configuration: pyyaml
- Machine learning: torch, transformers (optional)