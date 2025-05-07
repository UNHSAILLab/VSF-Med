#!/usr/bin/env python3
"""
Database Utilities for VSF-Med

This module provides database interaction utilities for the VSF-Med project,
including functions for retrieving, saving, and querying model responses.
"""

import os
import pandas as pd
import yaml
import re
from sqlalchemy import text
from sqlalchemy.engine import create_engine
from sqlalchemy.dialects.postgresql.base import PGDialect
from typing import Dict, Any, Optional, Union


def get_config_value(key_path: str, config_file: str) -> Any:
    """
    Load configuration value from YAML file.
    
    Args:
        key_path: Dot-separated path to the desired configuration value
        config_file: Path to the YAML configuration file
        
    Returns:
        The configuration value or None if not found
    """
    try:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)

        keys = key_path.split('.')
        value = data
        for key in keys:
            value = value[key]
        return value

    except FileNotFoundError:
        print(f"File {config_file} not found")
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
    except KeyError:
        print(f"Key path {key_path} not found")
    except Exception as e:
        print(f"Error: {e}")
    return None


def setup_database_connection(config_file: str = None, connection_string: str = None) -> Any:
    """
    Set up database connection based on configuration file or connection string.
    
    Args:
        config_file: Path to the YAML configuration file
        connection_string: SQLAlchemy connection string
        
    Returns:
        SQLAlchemy engine object
    """
    # Fix for PostgreSQL dialect versions
    def fake_get_server_version_info(self, connection):
        version_str = connection.execute(text("SELECT version()")).scalar()
        match = re.search(r'v(\d+)\.(\d+)\.(\d+)', version_str)
        if match:
            return tuple(map(int, match.groups()))
        return (13, 0, 0)
        
    PGDialect._get_server_version_info = fake_get_server_version_info
    
    # Determine connection string
    if connection_string:
        db_url = connection_string
    elif config_file:
        db_url = get_config_value("cd_url", config_file)
    else:
        raise ValueError("Either config_file or connection_string must be provided")
        
    # Create and return engine
    if db_url:
        return create_engine(db_url)
    else:
        raise ValueError("Could not determine database connection string")


def fetch_questions(engine, model_id: str = "gpt-4o", 
                   question_category: str = "original") -> pd.DataFrame:
    """
    Fetch questions from the database for model evaluation.
    
    Args:
        engine: SQLAlchemy database engine
        model_id: Model identifier for filtering
        question_category: Category of questions to retrieve
        
    Returns:
        DataFrame containing questions to process
    """
    query = text("""
    SELECT a.id, a.question_id, a.condition as question_type, a.text as question, 
           a.answer as ground_truth, a.image
    FROM mimicxp.mimic_all_qns a
    WHERE
    a.condition = :question_category
    """)
    
    return pd.read_sql(query, con=engine, params={"question_category": question_category})


def fetch_unprocessed_questions(engine, model_id: str, 
                              question_category: str = "original",
                              response_category: str = "visual_perturb") -> pd.DataFrame:
    """
    Fetch questions that have not yet been processed by a specific model.
    
    Args:
        engine: SQLAlchemy database engine
        model_id: Model identifier for filtering
        question_category: Category of questions to retrieve
        response_category: Category for model responses
        
    Returns:
        DataFrame containing unprocessed questions
    """
    query = text("""
    SELECT a.id, a.question_id, a.condition as question_type, a.text as question, 
           a.answer as ground_truth, a.image
    FROM mimicxp.mimic_all_qns a
    LEFT JOIN mimicxp.model_responses_r2 b
    ON CAST(a.question_id AS text) = b.question_id
    AND a.id = b.uid
    AND b.model_name = :model_id and
    a.condition = :question_category and
    b.question_category = :response_category
    WHERE b.question_id IS NULL
    """)
    
    return pd.read_sql(query, con=engine, params={
        "model_id": model_id,
        "question_category": question_category,
        "response_category": response_category
    })


def check_duplicate_response(engine, uid: str, question_id: Union[str, int], question: str, 
                           question_category: str, model_name: str, 
                           image_link: Optional[str] = None) -> bool:
    """
    Check if a response already exists in the database.
    
    Args:
        engine: SQLAlchemy database engine
        uid: Unique identifier
        question_id: Question identifier
        question: Question text
        question_category: Category of the response
        model_name: Name of the model
        image_link: Optional image link for checking
        
    Returns:
        True if a duplicate exists, False otherwise
    """
    query = text("""
        SELECT 1 FROM mimicxp.model_responses_r2
        WHERE
        uid = :uid
        AND question_id = :question_id and
        question = :question
          AND question_category = :question_category
          AND model_name = :model_name
        LIMIT 1
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {
            "uid": uid,
            "question_id": question_id,
            "question": question,
            "question_category": question_category,
            "model_name": model_name
        }).fetchone()
        
    return result is not None


def insert_model_response(engine, uid: str, question_id: Union[str, int], question: str, 
                        question_category: str, actual_answer: str, model_name: str, 
                        model_answer: str, image_link: Optional[str] = None) -> bool:
    """
    Insert a model's response into the database.
    
    Args:
        engine: SQLAlchemy database engine
        uid: Unique identifier
        question_id: Question identifier
        question: Question text
        question_category: Category of the response
        actual_answer: Ground truth answer
        model_name: Name of the model
        model_answer: Model's response
        image_link: Optional image link
        
    Returns:
        True if insertion was successful, False otherwise
    """
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text("""
                INSERT INTO mimicxp.model_responses_r2
                (uid, question_id, question, question_category, actual_answer, model_name, model_answer, image_link)
                VALUES (:uid, :question_id, :question, :question_category, :actual_answer, :model_name, :model_answer, :image_link)
            """), {
                "uid": uid,
                "question_id": question_id,
                "question": question,
                "question_category": question_category,
                "actual_answer": actual_answer,
                "model_name": model_name,
                "model_answer": model_answer,
                "image_link": image_link
            })
            trans.commit()
            return True
            
        except Exception as e:
            trans.rollback()
            print(f"Error inserting response: {str(e)}")
            return False


def get_model_responses(engine, model_name: str, 
                      question_category: str = "visual_perturb",
                      limit: int = 100) -> pd.DataFrame:
    """
    Retrieve model responses from the database.
    
    Args:
        engine: SQLAlchemy database engine
        model_name: Name of the model
        question_category: Category of responses to retrieve
        limit: Maximum number of responses to retrieve
        
    Returns:
        DataFrame containing model responses
    """
    query = text("""
    SELECT uid, question_id, question, question_category, 
           actual_answer, model_name, model_answer, image_link
    FROM mimicxp.model_responses_r2
    WHERE model_name = :model_name
      AND question_category = :question_category
    ORDER BY question_id
    LIMIT :limit
    """)
    
    return pd.read_sql(query, con=engine, params={
        "model_name": model_name,
        "question_category": question_category,
        "limit": limit
    })


def get_evaluation_results(engine, model_name: str, 
                         question_category: str = "visual_perturb",
                         limit: int = 100) -> pd.DataFrame:
    """
    Retrieve evaluation results from the database.
    
    Args:
        engine: SQLAlchemy database engine
        model_name: Name of the model
        question_category: Category of responses to retrieve
        limit: Maximum number of results to retrieve
        
    Returns:
        DataFrame containing evaluation results
    """
    query = text("""
    SELECT e.id, e.model_response_id, r.question, r.model_answer,
           e.prompt_injection_score, e.jailbreak_score, e.confidentiality_score,
           e.misinformation_score, e.dos_resilience_score, e.persistence_score,
           e.safety_bypass_score, e.medical_impact_score,
           e.total_score, e.risk_classification, e.remarks
    FROM mimicxp.evaluations e
    JOIN mimicxp.model_responses_r2 r ON e.model_response_id = r.id
    WHERE r.model_name = :model_name
      AND r.question_category = :question_category
    ORDER BY e.total_score DESC
    LIMIT :limit
    """)
    
    return pd.read_sql(query, con=engine, params={
        "model_name": model_name,
        "question_category": question_category,
        "limit": limit
    })


def save_evaluation(engine, model_response_id: int, scores: Dict[str, Any]) -> bool:
    """
    Save evaluation scores for a model response.
    
    Args:
        engine: SQLAlchemy database engine
        model_response_id: ID of the model response
        scores: Dictionary of vulnerability scores
        
    Returns:
        True if insertion was successful, False otherwise
    """
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text("""
                INSERT INTO mimicxp.evaluations
                (model_response_id, prompt_injection_score, jailbreak_score, confidentiality_score,
                 misinformation_score, dos_resilience_score, persistence_score,
                 safety_bypass_score, medical_impact_score,
                 total_score, risk_classification, remarks)
                VALUES (:model_response_id, :prompt_injection_score, :jailbreak_score, :confidentiality_score,
                        :misinformation_score, :dos_resilience_score, :persistence_score,
                        :safety_bypass_score, :medical_impact_score,
                        :total_score, :risk_classification, :remarks)
            """), {
                "model_response_id": model_response_id,
                "prompt_injection_score": scores.get("Prompt Injection Effectiveness", 0),
                "jailbreak_score": scores.get("Jailbreak Success", 0),
                "confidentiality_score": scores.get("Confidentiality Breach", 0),
                "misinformation_score": scores.get("Misinformation or Malicious Content Generation", 0),
                "dos_resilience_score": scores.get("Denial-of-Service Attack Resilience", 0),
                "persistence_score": scores.get("Persistence of Attack Effects", 0),
                "safety_bypass_score": scores.get("Safety Bypass Success", 0),
                "medical_impact_score": scores.get("Impact on Medical Decision Support", 0),
                "total_score": scores.get("total_score", 0),
                "risk_classification": scores.get("Severity Classification", "Unknown"),
                "remarks": scores.get("remarks", "")
            })
            trans.commit()
            return True
            
        except Exception as e:
            trans.rollback()
            print(f"Error saving evaluation: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="VSF-Med Database Utilities")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--action", type=str, choices=["list", "fetch", "stats"], 
                        default="list", help="Action to perform")
    
    args = parser.parse_args()
    
    # Set up database connection
    engine = setup_database_connection(config_file=args.config)
    
    if args.action == "list":
        # List model responses
        responses = get_model_responses(engine, args.model)
        print(f"Found {len(responses)} responses for model {args.model}")
        print(responses.head())
        
    elif args.action == "fetch":
        # Fetch unprocessed questions
        questions = fetch_unprocessed_questions(engine, args.model)
        print(f"Found {len(questions)} unprocessed questions for model {args.model}")
        print(questions.head())
        
    elif args.action == "stats":
        # Show evaluation statistics
        evaluations = get_evaluation_results(engine, args.model)
        print(f"Found {len(evaluations)} evaluations for model {args.model}")
        
        if not evaluations.empty:
            print(f"Average total score: {evaluations['total_score'].mean():.2f}")
            print(f"Risk classifications:")
            print(evaluations['risk_classification'].value_counts())
            
            print("\nAverage scores by dimension:")
            for col in ["prompt_injection_score", "jailbreak_score", "confidentiality_score",
                       "misinformation_score", "dos_resilience_score", "persistence_score",
                       "safety_bypass_score", "medical_impact_score"]:
                print(f"- {col}: {evaluations[col].mean():.2f}")