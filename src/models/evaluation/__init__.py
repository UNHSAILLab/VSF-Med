"""
Evaluation models for VSF-Med.

This package provides models and utilities for evaluating
the vulnerability of medical vision-language models.
"""

from src.models.evaluation.vulnerability_scoring import VulnerabilityScorer

__all__ = ['VulnerabilityScorer']