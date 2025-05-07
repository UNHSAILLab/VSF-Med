"""
Visualization utilities for VSF-Med.

This package provides utilities for visualizing and analyzing
medical images and model evaluation results.
"""

from src.utils.visualization.image_utils import (
    load_image,
    display_image,
    compare_images,
    compute_similarity_metrics,
    highlight_differences,
    visualize_comparison,
    create_perturbation_grid
)

__all__ = [
    'load_image',
    'display_image',
    'compare_images',
    'compute_similarity_metrics',
    'highlight_differences',
    'visualize_comparison',
    'create_perturbation_grid'
]