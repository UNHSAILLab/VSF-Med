-- VSF-Med Database Schema

-- Schema creation
CREATE SCHEMA IF NOT EXISTS mimicxp;

-- Questions table - stores all questions for the experiments
CREATE TABLE IF NOT EXISTS mimicxp.mimic_all_qns (
    id VARCHAR(255) NOT NULL,  -- Source image ID (e.g., DICOM identifier)
    question_id SERIAL PRIMARY KEY,
    condition VARCHAR(50) NOT NULL,  -- Type of question: 'original', 'adversarial', etc.
    text TEXT NOT NULL,  -- The question text
    answer TEXT,  -- Ground truth answer (if available)
    image VARCHAR(255) NOT NULL,  -- Relative path to image file in MIMIC-CXR dataset
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_mimic_all_qns_id ON mimicxp.mimic_all_qns(id);
CREATE INDEX IF NOT EXISTS idx_mimic_all_qns_condition ON mimicxp.mimic_all_qns(condition);

-- Model responses table - stores responses from various models
CREATE TABLE IF NOT EXISTS mimicxp.model_responses_r2 (
    id SERIAL PRIMARY KEY,
    uid VARCHAR(255) NOT NULL,  -- Source image ID
    question_id VARCHAR(50) NOT NULL,  -- Question ID
    question TEXT NOT NULL,  -- The original question
    question_category VARCHAR(50) NOT NULL,  -- 'original', 'visual_perturb', 'text_attack', etc.
    actual_answer TEXT,  -- Ground truth answer (if available)
    model_name VARCHAR(100) NOT NULL,  -- Model identifier (e.g., 'gpt-4o', 'gemma-3-4b-it')
    model_answer TEXT NOT NULL,  -- Model's response
    image_link TEXT,  -- Path to the image file used
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_model_responses_uid ON mimicxp.model_responses_r2(uid);
CREATE INDEX IF NOT EXISTS idx_model_responses_question_id ON mimicxp.model_responses_r2(question_id);
CREATE INDEX IF NOT EXISTS idx_model_responses_model_name ON mimicxp.model_responses_r2(model_name);
CREATE INDEX IF NOT EXISTS idx_model_responses_category ON mimicxp.model_responses_r2(question_category);

-- Evaluations table - stores vulnerability assessments
CREATE TABLE IF NOT EXISTS mimicxp.evaluations (
    id SERIAL PRIMARY KEY,
    model_response_id INTEGER REFERENCES mimicxp.model_responses_r2(id),
    prompt_injection_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    jailbreak_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    confidentiality_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    misinformation_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    dos_resilience_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    persistence_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    safety_bypass_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    medical_impact_score INTEGER NOT NULL DEFAULT 0, -- 0-4 scale
    total_score INTEGER NOT NULL DEFAULT 0, -- Sum of all scores (0-32)
    risk_classification VARCHAR(50) NOT NULL, -- Low, Moderate, High, Critical
    remarks TEXT, -- Additional notes or observations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_evaluations_model_response_id ON mimicxp.evaluations(model_response_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_total_score ON mimicxp.evaluations(total_score);
CREATE INDEX IF NOT EXISTS idx_evaluations_risk ON mimicxp.evaluations(risk_classification);

-- Perturbation metadata table - stores information about perturbations
CREATE TABLE IF NOT EXISTS mimicxp.perturbation_metadata (
    id SERIAL PRIMARY KEY,
    original_image_path VARCHAR(255) NOT NULL,
    perturbed_image_path VARCHAR(255) NOT NULL,
    perturbation_type VARCHAR(100) NOT NULL, -- 'gaussian_noise', 'checkerboard', etc.
    parameters JSONB NOT NULL, -- Stores perturbation parameters
    ssim FLOAT, -- Structural Similarity Index
    psnr FLOAT, -- Peak Signal-to-Noise Ratio
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_perturbation_original ON mimicxp.perturbation_metadata(original_image_path);
CREATE INDEX IF NOT EXISTS idx_perturbation_type ON mimicxp.perturbation_metadata(perturbation_type);

-- View for aggregating evaluation statistics by model and question category
CREATE OR REPLACE VIEW mimicxp.model_vulnerability_stats AS
SELECT 
    r.model_name,
    r.question_category,
    COUNT(e.id) AS num_evaluations,
    AVG(e.total_score) AS avg_total_score,
    AVG(e.prompt_injection_score) AS avg_prompt_injection,
    AVG(e.jailbreak_score) AS avg_jailbreak,
    AVG(e.confidentiality_score) AS avg_confidentiality,
    AVG(e.misinformation_score) AS avg_misinformation,
    AVG(e.dos_resilience_score) AS avg_dos_resilience,
    AVG(e.persistence_score) AS avg_persistence,
    AVG(e.safety_bypass_score) AS avg_safety_bypass,
    AVG(e.medical_impact_score) AS avg_medical_impact,
    COUNT(CASE WHEN e.risk_classification = 'Low Risk' THEN 1 END) AS low_risk_count,
    COUNT(CASE WHEN e.risk_classification = 'Moderate Risk' THEN 1 END) AS moderate_risk_count,
    COUNT(CASE WHEN e.risk_classification = 'High Risk' THEN 1 END) AS high_risk_count,
    COUNT(CASE WHEN e.risk_classification = 'Critical Risk' THEN 1 END) AS critical_risk_count
FROM 
    mimicxp.evaluations e
JOIN 
    mimicxp.model_responses_r2 r ON e.model_response_id = r.id
GROUP BY 
    r.model_name, r.question_category;

-- Sample query to get all questions and their perturbations
-- SELECT q.question_id, q.text, q.condition, r.question_category, r.model_name, r.model_answer, p.perturbation_type, p.ssim
-- FROM mimicxp.mimic_all_qns q
-- JOIN mimicxp.model_responses_r2 r ON q.question_id::text = r.question_id
-- LEFT JOIN mimicxp.perturbation_metadata p ON r.image_link = p.perturbed_image_path
-- WHERE q.condition = 'original' AND r.question_category = 'visual_perturb'
-- ORDER BY q.question_id;