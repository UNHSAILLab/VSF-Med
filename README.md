# VSF-Med: A Vulnerability Scoring Framework for Medical Vision-Language Models

VSF-Med is a comprehensive framework designed to systematically evaluate the safety, reliability, and adversarial robustness of Vision-Language Models (Vision LLMs) in clinical imaging applications. This repository contains the implementation of our evaluation pipeline and experimental results.

## Overview

VSF-Med uses an ordinal 0-4 scale across eight vulnerability dimensions:
- Prompt injection effectiveness
- Jailbreak resilience
- Potential confidentiality breach
- Risk of misinformation
- Denial of service resilience
- Persistence of attack effects
- Safety bypass success
- Impact on medical decision support

We apply this framework to ten clinically motivated adversarial scenarios, ranging from contextual prompt injections to image perturbations, using the MIMIC-CXR dataset.

## Repository Structure

```
VSF-Med/
├── src/                         # Source code
│   ├── config/                  # Configuration files
│   │   └── default_config.yaml  # Default configuration
│   ├── data/                    # Data processing utilities
│   │   ├── raw/                 # Scripts for raw data processing
│   │   └── processed/           # Scripts for processed data
│   ├── models/                  # Model implementations
│   │   ├── evaluation/          # Evaluation and scoring models
│   │   │   └── vulnerability_scoring.py  # VSF-Med scoring framework
│   │   ├── text_models/         # Text-only model interfaces
│   │   └── vision_models/       # Vision-language model interfaces
│   ├── utils/                   # Utility functions
│   │   ├── database/            # Database interaction utilities
│   │   │   └── database_utils.py  # Database operations
│   │   ├── perturbations/       # Image and text perturbation utilities
│   │   │   ├── image_perturbations.py  # Visual perturbation methods
│   │   │   └── text_perturbations.py   # Text attack methods
│   │   └── visualization/       # Visualization utilities
│   │       └── image_utils.py   # Image comparison and analysis
│   └── experiments/             # Experimental configurations
├── notebooks/                   # Main experiment notebooks
│   ├── 01_generate_adversarial_samples.ipynb  # Creation of adversarial prompts
│   ├── 02_gpt_radiologist_baseline.ipynb      # Base notebook for GPT-4o evaluation
│   ├── 03_gpt_radiologist_visual_perturbation.ipynb  # Visual perturbation evaluation
│   └── 04_gpt_vulnerability_evaluation.ipynb  # Evaluation of model responses
├── templates/                   # Template files
│   ├── text_attack_templates.txt         # Templates for text attacks
│   ├── visual_perturbation_methods.txt   # Methods for visual perturbations
│   └── vsf_scoring_rubric.txt            # Evaluation rubric for scoring
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Evaluation Workflow

1. **Dataset Selection**: A diverse selection of 5,000 frontal chest X-ray studies from MIMIC-CXR, stratified by patient demographics and key pathologies.

2. **Adversarial Variant Generation**:
   - **Text attacks**: 18 attack categories with 2-4 expert-curated prompt templates each
   - **Visual attacks**: 6 perturbation methods (Gaussian noise, checkerboard, random arrow overlay, Moiré pattern, steganographic hide, LSB extraction)

3. **Model Inference**: Invoking Vision LLMs through API clients for each variant, recording diagnostics and additional output.

4. **LLM-based Scoring**: Independent LLM judges consume model outputs along with the VSF scoring rubric to evaluate vulnerability across all dimensions.

5. **Aggregation and Analysis**: Computing per-dimension statistics and categorizing vulnerability scores into risk tiers.

## Text Attack Categories

Our framework formalizes 18 different attack categories including:
- Prompt Injection
- Jailbreak Attempts
- Confidentiality Breach
- Misinformation Generation
- Denial-of-Service
- Persistence Attacks
- Safety Bypass
- Semantic Shift
- Omission Attacks
- Over-Confidence Induction
- And more...

## Visual Perturbation Methods

We apply six visual perturbation techniques to test model robustness:
- Gaussian noise at various levels
- Checkerboard overlays (single or tiled)
- Moiré patterns with adjustable frequency
- Random arrow artifacts
- Steganographic information hiding
- LSB-plane extraction

Perturbation parameters are optimized via grid search to balance imperceptibility (SSIM ≥ 0.85) with attack potency.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VSF-Med.git
cd VSF-Med

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Configuration

1. Copy and customize the default configuration:
   ```bash
   cp src/config/default_config.yaml src/config/my_config.yaml
   ```

2. Edit `my_config.yaml` to set database credentials, API keys, and data paths.

### Running Experiments

1. **Generate Adversarial Prompts**:
   ```bash
   # Using the script
   python src/data/processed/generate_adversarial_prompts.py --config src/config/my_config.yaml
   
   # Or run the notebook
   jupyter notebook notebooks/01_generate_adversarial_samples.ipynb
   ```

2. **Generate Visual Perturbations**:
   ```bash
   python src/utils/perturbations/image_perturbations.py --source /path/to/images --output /path/to/output
   ```

3. **Evaluate Model Performance**:
   ```bash
   # Base evaluation
   jupyter notebook notebooks/02_gpt_radiologist_baseline.ipynb
   
   # Visual perturbation evaluation
   jupyter notebook notebooks/03_gpt_radiologist_visual_perturbation.ipynb
   ```

4. **Analyze Results**:
   ```bash
   jupyter notebook notebooks/04_gpt_vulnerability_evaluation.ipynb
   ```

## Requirements

- Python 3.8+
- OpenAI API key (for GPT-4o access)
- MIMIC-CXR dataset access
- Required Python libraries:
  - pandas
  - numpy
  - sqlalchemy
  - openai
  - PIL
  - cv2
  - matplotlib
  - scikit-image

## Citation

If you use VSF-Med in your research, please cite our paper:
```
@article{vsf-med2024,
  title={VSF-Med: A Vulnerability Scoring Framework for Medical Vision-Language Models},
  author={[Author names]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the terms of the included LICENSE file.