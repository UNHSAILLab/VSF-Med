"""
Database utilities for VSF-Med.

This package provides utilities for interacting with databases
for storing and retrieving model data and evaluations.
"""

from src.utils.database.database_utils import (
    get_config_value,
    setup_database_connection,
    fetch_questions,
    fetch_unprocessed_questions,
    check_duplicate_response,
    insert_model_response,
    get_model_responses,
    get_evaluation_results,
    save_evaluation
)

__all__ = [
    'get_config_value',
    'setup_database_connection',
    'fetch_questions',
    'fetch_unprocessed_questions',
    'check_duplicate_response',
    'insert_model_response',
    'get_model_responses',
    'get_evaluation_results',
    'save_evaluation'
]