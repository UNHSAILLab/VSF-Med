#!/usr/bin/env python3
"""
Image Visualization and Processing Utilities for VSF-Med

This module provides functions for visualizing and analyzing medical images,
including comparison between original and perturbed images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Tuple, Optional, Union, List, Dict, Any


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as NumPy array
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be read
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
        
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    return image


def display_image(image: Union[str, np.ndarray], title: str = "Image", 
                 figsize: Tuple[int, int] = (6, 6), 
                 cmap: Optional[str] = None) -> None:
    """
    Display an image for visual inspection.
    
    Args:
        image: Image path or NumPy array
        title: Title for the plot
        figsize: Figure size as (width, height)
        cmap: Colormap to use (default: grayscale for 1-channel images)
    """
    try:
        if isinstance(image, str):
            img = load_image(image)
        else:
            img = image.copy()
            
        # Determine colormap
        if cmap is None:
            cmap = 'gray' if len(img.shape) == 2 else None
            
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying image: {str(e)}")


def compare_images(original: Union[str, np.ndarray], perturbed: Union[str, np.ndarray], 
                  titles: Tuple[str, str] = ("Original", "Perturbed"),
                  figsize: Tuple[int, int] = (12, 6), 
                  cmap: Optional[str] = None) -> None:
    """
    Display two images side by side for comparison.
    
    Args:
        original: Original image path or array
        perturbed: Perturbed image path or array
        titles: Tuple of titles for (original, perturbed)
        figsize: Figure size as (width, height)
        cmap: Colormap to use
    """
    try:
        # Load images if paths provided
        if isinstance(original, str):
            orig_img = load_image(original)
        else:
            orig_img = original.copy()
            
        if isinstance(perturbed, str):
            pert_img = load_image(perturbed)
        else:
            pert_img = perturbed.copy()
            
        # Determine colormap
        if cmap is None:
            cmap = 'gray' if len(orig_img.shape) == 2 else None
            
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Show original image
        axes[0].imshow(orig_img, cmap=cmap)
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        
        # Show perturbed image
        axes[1].imshow(pert_img, cmap=cmap)
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error comparing images: {str(e)}")


def compute_similarity_metrics(original: np.ndarray, perturbed: np.ndarray) -> Dict[str, float]:
    """
    Compute various similarity metrics between original and perturbed images.
    
    Args:
        original: Original image as NumPy array
        perturbed: Perturbed image as NumPy array
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    try:
        # Ensure images have the same dimensions
        if original.shape != perturbed.shape:
            raise ValueError(f"Images have different shapes: {original.shape} vs {perturbed.shape}")
            
        # Mean Squared Error (MSE)
        mse = np.mean((original - perturbed) ** 2)
        metrics["MSE"] = float(mse)
        
        # Peak Signal-to-Noise Ratio (PSNR)
        if mse > 0:
            max_pixel = 255.0
            metrics["PSNR"] = float(20 * np.log10(max_pixel / np.sqrt(mse)))
        else:
            metrics["PSNR"] = float('inf')
            
        # Mean Absolute Error (MAE)
        metrics["MAE"] = float(np.mean(np.abs(original - perturbed)))
        
        # Structural Similarity Index (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Handle multichannel images
            if len(original.shape) == 3 and original.shape[2] == 3:
                ssim_value = ssim(original, perturbed, channel_axis=2, data_range=255)
            else:
                ssim_value = ssim(original, perturbed, data_range=255)
                
            metrics["SSIM"] = float(ssim_value)
            
        except ImportError:
            metrics["SSIM"] = None
            print("Warning: skimage.metrics.structural_similarity not available")
            
    except Exception as e:
        print(f"Error computing similarity metrics: {str(e)}")
        
    return metrics


def highlight_differences(original: np.ndarray, perturbed: np.ndarray, 
                         threshold: float = 0.1) -> np.ndarray:
    """
    Create a difference map highlighting areas of perturbation.
    
    Args:
        original: Original image as NumPy array
        perturbed: Perturbed image as NumPy array
        threshold: Threshold for highlighting differences (0-1)
        
    Returns:
        Difference map image as NumPy array
    """
    try:
        # Ensure images have the same dimensions
        if original.shape != perturbed.shape:
            raise ValueError(f"Images have different shapes: {original.shape} vs {perturbed.shape}")
            
        # Convert to grayscale if needed
        if len(original.shape) == 3 and original.shape[2] == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            pert_gray = cv2.cvtColor(perturbed, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            pert_gray = perturbed
            
        # Compute absolute difference
        diff = cv2.absdiff(orig_gray, pert_gray)
        
        # Normalize difference to 0-1 range
        diff_norm = diff / 255.0
        
        # Apply threshold
        mask = diff_norm > threshold
        
        # Create heatmap image (red highlights)
        heatmap = np.zeros((*orig_gray.shape, 3), dtype=np.uint8)
        heatmap[mask, 0] = 255  # Red channel
        
        # Create overlay by blending original image with heatmap
        if len(original.shape) == 3 and original.shape[2] == 3:
            result = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)
        else:
            # Convert grayscale to RGB for overlay
            orig_rgb = np.stack([orig_gray] * 3, axis=2)
            result = cv2.addWeighted(orig_rgb, 0.7, heatmap, 0.3, 0)
            
        return result
        
    except Exception as e:
        print(f"Error highlighting differences: {str(e)}")
        return original.copy()  # Return original if error occurs


def visualize_comparison(original: Union[str, np.ndarray], perturbed: Union[str, np.ndarray],
                        figsize: Tuple[int, int] = (15, 5)) -> Dict[str, float]:
    """
    Comprehensive visualization of original vs perturbed image with metrics.
    
    Args:
        original: Original image path or array
        perturbed: Perturbed image path or array
        figsize: Figure size as (width, height)
        
    Returns:
        Dictionary of computed similarity metrics
    """
    try:
        # Load images if paths provided
        if isinstance(original, str):
            orig_img = load_image(original)
        else:
            orig_img = original.copy()
            
        if isinstance(perturbed, str):
            pert_img = load_image(perturbed)
        else:
            pert_img = perturbed.copy()
            
        # Compute metrics
        metrics = compute_similarity_metrics(orig_img, pert_img)
        
        # Create difference visualization
        diff_img = highlight_differences(orig_img, pert_img)
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Determine colormap
        cmap = 'gray' if len(orig_img.shape) == 2 else None
        
        # Show original image
        axes[0].imshow(orig_img, cmap=cmap)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Show perturbed image
        axes[1].imshow(pert_img, cmap=cmap)
        ssim_val = metrics.get("SSIM", "N/A")
        psnr_val = metrics.get("PSNR", "N/A")
        axes[1].set_title(f"Perturbed (SSIM: {ssim_val:.3f}, PSNR: {psnr_val:.1f})")
        axes[1].axis('off')
        
        # Show difference map
        axes[2].imshow(diff_img)
        axes[2].set_title("Difference Map")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return metrics
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return {}


def create_perturbation_grid(original: np.ndarray, 
                           perturbation_function: callable, 
                           param_name: str,
                           param_values: List[Any], 
                           figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Create a grid of perturbed images with varying parameter values.
    
    Args:
        original: Original image as NumPy array
        perturbation_function: Function that takes (image, param_value) and returns perturbed image
        param_name: Name of the parameter being varied
        param_values: List of parameter values to test
        figsize: Figure size as (width, height)
    """
    try:
        n_values = len(param_values)
        n_cols = 3  # Number of columns in the grid
        n_rows = (n_values + n_cols) // n_cols  # Calculate number of rows needed
        
        # Create figure
        fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Display original image in first cell
        cmap = 'gray' if len(original.shape) == 2 else None
        axes[0].imshow(original, cmap=cmap)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Generate and display perturbed images
        for i, value in enumerate(param_values):
            # Apply perturbation
            perturbed = perturbation_function(original, value)
            
            # Calculate metrics
            metrics = compute_similarity_metrics(original, perturbed)
            ssim_val = metrics.get("SSIM", "N/A")
            
            # Display perturbed image
            axes[i+1].imshow(perturbed, cmap=cmap)
            axes[i+1].set_title(f"{param_name}={value}\nSSIM={ssim_val:.3f}")
            axes[i+1].axis('off')
            
        # Hide any unused subplots
        for i in range(n_values + 1, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.suptitle(f"Effect of {param_name} on Image Perturbation", fontsize=16, y=1.02)
        plt.show()
        
    except Exception as e:
        print(f"Error creating perturbation grid: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Visualization Utilities")
    parser.add_argument("--original", type=str, required=True, help="Path to original image")
    parser.add_argument("--perturbed", type=str, help="Path to perturbed image")
    parser.add_argument("--output", type=str, help="Path for saving visualization output")
    
    args = parser.parse_args()
    
    if args.perturbed:
        # Run comparison visualization
        metrics = visualize_comparison(args.original, args.perturbed)
        print("Similarity Metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value}")
    else:
        # Just display the original image
        display_image(args.original, title="Image")