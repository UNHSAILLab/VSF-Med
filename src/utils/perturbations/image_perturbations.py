#!/usr/bin/env python3
"""
Image Perturbation Module for VSF-Med

This module provides various image perturbation techniques to evaluate 
the robustness of medical vision-language models against visual adversarial attacks.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, UnidentifiedImageError
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, Dict, Any

class ImagePerturbation:
    """
    Class implementing various image perturbation techniques for medical imaging.
    """
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0, stddev: float = 25) -> np.ndarray:
        """
        Add Gaussian noise to an image.
        
        Args:
            image: Input image as NumPy array
            mean: Mean of the Gaussian noise
            stddev: Standard deviation of the noise
            
        Returns:
            Image with added Gaussian noise
        """
        img_float = image.astype(np.float32)
        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_img = img_float + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img
    
    @staticmethod
    def add_salt_and_pepper(image: np.ndarray, prob: float = 0.01) -> np.ndarray:
        """
        Add salt and pepper noise to an image.
        
        Args:
            image: Input image as NumPy array
            prob: Probability of pixel alteration
            
        Returns:
            Image with salt and pepper noise
        """
        noisy_img = image.copy()
        rnd = np.random.rand(*image.shape[:2])
        salt = rnd < (prob / 2)
        pepper = (rnd >= (prob / 2)) & (rnd < prob)
        
        if len(image.shape) == 2:  # Grayscale
            noisy_img[salt] = 255
            noisy_img[pepper] = 0
        else:  # Color image
            noisy_img[salt] = [255, 255, 255]
            noisy_img[pepper] = [0, 0, 0]
            
        return noisy_img
    
    @staticmethod
    def add_uniform_noise(image: np.ndarray, low: float = -25, high: float = 25) -> np.ndarray:
        """
        Add uniform noise to an image.
        
        Args:
            image: Input image as NumPy array
            low: Lower bound for noise values
            high: Upper bound for noise values
            
        Returns:
            Image with uniform noise
        """
        img_float = image.astype(np.float32)
        noise = np.random.uniform(low, high, image.shape).astype(np.float32)
        noisy_img = img_float + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, ksize: Tuple[int, int] = (5, 5), sigmaX: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur to an image.
        
        Args:
            image: Input image as NumPy array
            ksize: Kernel size (must be odd numbers)
            sigmaX: Gaussian kernel standard deviation in X direction
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, ksize, sigmaX)
    
    @staticmethod
    def apply_average_blur(image: np.ndarray, ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Apply average blur to an image.
        
        Args:
            image: Input image as NumPy array
            ksize: Kernel size
            
        Returns:
            Blurred image
        """
        return cv2.blur(image, ksize)
    
    @staticmethod
    def apply_median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Apply median blur to an image.
        
        Args:
            image: Input image as NumPy array
            ksize: Kernel size (must be odd)
            
        Returns:
            Blurred image
        """
        return cv2.medianBlur(image, ksize)
    
    @staticmethod
    def adjust_contrast_brightness(image: np.ndarray, alpha: float = 1.5, beta: int = 30) -> np.ndarray:
        """
        Adjust contrast and brightness of an image.
        
        Args:
            image: Input image as NumPy array
            alpha: Contrast control (1.0 means no change)
            beta: Brightness control (0 means no change)
            
        Returns:
            Adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def simulate_compression_artifacts(image: np.ndarray, quality: int = 30) -> np.ndarray:
        """
        Simulate JPEG compression artifacts.
        
        Args:
            image: Input image as NumPy array
            quality: JPEG quality factor (0-100, lower means more artifacts)
            
        Returns:
            Image with compression artifacts
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        
        if result:
            return cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Failed to encode image for compression simulation")
    
    @staticmethod
    def make_checkerboard(patch_size: int = 100, square_size: int = 25, fill: int = 128) -> Image.Image:
        """
        Create a checkerboard pattern.
        
        Args:
            patch_size: Size of the pattern in pixels
            square_size: Size of each square in pixels
            fill: Gray level for filled squares (0-255)
            
        Returns:
            PIL Image with checkerboard pattern
        """
        p = Image.new("L", (patch_size, patch_size), 0)
        draw = ImageDraw.Draw(p)
        
        for y in range(0, patch_size, 2 * square_size):
            for x in range(0, patch_size, 2 * square_size):
                draw.rectangle([x, y, x + square_size - 1, y + square_size - 1], fill=fill)
                draw.rectangle(
                    [x + square_size, y + square_size, x + 2 * square_size - 1, y + 2 * square_size - 1], 
                    fill=fill
                )
                
        return p
    
    @staticmethod
    def overlay_checkerboard(image: np.ndarray, patch: Image.Image, tiled: bool = False) -> np.ndarray:
        """
        Overlay a checkerboard pattern on an image.
        
        Args:
            image: Input image as NumPy array
            patch: Checkerboard pattern as PIL Image
            tiled: Whether to tile the pattern or place single instance
            
        Returns:
            Image with overlaid checkerboard
        """
        # Convert NumPy array to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
        else:
            pil_img = image
            
        out = pil_img.copy()
        w, h = pil_img.size
        pw, ph = patch.size
        
        if tiled:
            # Tile patches every (pw+gap) pixels
            gap = pw // 4
            for y in range(0, h, ph + gap):
                for x in range(0, w, pw + gap):
                    out.paste(patch, (x, y), patch)
        else:
            # Single random placement
            x = np.random.randint(0, max(1, w - pw))
            y = np.random.randint(0, max(1, h - ph))
            out.paste(patch, (x, y), patch)
        
        # Convert back to NumPy array
        result = np.array(out)
        if len(result.shape) == 3 and result.shape[2] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
        return result
    
    @staticmethod
    def add_random_arrow(image: np.ndarray) -> np.ndarray:
        """
        Add a random arrow to an image.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Image with added arrow
        """
        # Convert NumPy array to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
        else:
            pil_img = image
            
        w, h = pil_img.size
        start = (np.random.randint(w), np.random.randint(h))
        end = (np.random.randint(w), np.random.randint(h))
        
        out = pil_img.copy().convert("RGB")
        d = ImageDraw.Draw(out)
        d.line([start, end], width=3, fill=(255,255,255))
        
        dx, dy = end[0]-start[0], end[1]-start[1]
        perp = (-dy, dx)
        norm = np.hypot(*perp)
        
        if norm > 0:
            ux, uy = perp[0]/norm*10, perp[1]/norm*10
            head = [
                end,
                (int(end[0] - 0.2*dx + ux), int(end[1] - 0.2*dy + uy)),
                (int(end[0] - 0.2*dx - ux), int(end[1] - 0.2*dy - uy))
            ]
            d.polygon(head, fill=(255,255,255))
            
        result = np.array(out.convert("L"))
        return result
    
    @staticmethod
    def make_moire_pattern(size: Tuple[int, int], freq: float = 0.1) -> np.ndarray:
        """
        Create a Moiré pattern.
        
        Args:
            size: Size of the pattern as (width, height)
            freq: Frequency of the pattern
            
        Returns:
            NumPy array with Moiré pattern
        """
        w, h = size
        xs = np.linspace(0, 2*np.pi*freq*w, w)
        ys = np.linspace(0, 2*np.pi*freq*h, h)
        grid = np.outer(np.sin(xs), np.sin(ys))
        norm = ((grid+1)/2 * 255).astype(np.uint8)
        return norm
    
    @staticmethod
    def overlay_moire_pattern(image: np.ndarray, freq: float = 0.1, alpha: float = 0.3) -> np.ndarray:
        """
        Overlay a Moiré pattern on an image.
        
        Args:
            image: Input image as NumPy array
            freq: Frequency of the pattern
            alpha: Blend factor (0.0-1.0)
            
        Returns:
            Image with overlaid Moiré pattern
        """
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            moire = ImagePerturbation.make_moire_pattern((w, h), freq)
            
            # Ensure moire pattern matches image dimensions
            if len(image.shape) == 3:  # Color image
                moire = np.stack([moire] * 3, axis=2)
                
            # Blend images
            result = cv2.addWeighted(image, 1-alpha, moire, alpha, 0)
            return result
        else:
            raise ValueError("Input must be a NumPy array")
    
    @staticmethod
    def display_image(image: Union[str, np.ndarray], title: str = "Image", figsize: Tuple[int, int] = (6, 6)) -> None:
        """
        Display an image for visual inspection.
        
        Args:
            image: Image path or NumPy array
            title: Title for the plot
            figsize: Figure size as (width, height)
        """
        try:
            if isinstance(image, str):
                img = plt.imread(image)
            else:
                img = image
                
            if len(img.shape) == 3 and img.shape[2] == 3:
                if img.max() > 1 and isinstance(image, np.ndarray):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
            plt.figure(figsize=figsize)
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            plt.title(title)
            plt.axis('off')
            plt.show()
            
        except FileNotFoundError:
            print(f"Error: Image file not found at {image}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    @classmethod
    def perturb_image(cls, image: np.ndarray, technique: str = "gaussian_noise", **kwargs) -> np.ndarray:
        """
        Apply a perturbation technique to an image.
        
        Args:
            image: Input image as NumPy array
            technique: Perturbation technique to apply
            **kwargs: Additional parameters for the specific technique
            
        Returns:
            Perturbed image
            
        Raises:
            ValueError: If an unknown technique is specified
        """
        perturbation_methods = {
            "gaussian_noise": cls.add_gaussian_noise,
            "salt_and_pepper": cls.add_salt_and_pepper,
            "uniform_noise": cls.add_uniform_noise,
            "gaussian_blur": cls.apply_gaussian_blur,
            "average_blur": cls.apply_average_blur,
            "median_blur": cls.apply_median_blur,
            "contrast_brightness": cls.adjust_contrast_brightness,
            "compression_artifact": cls.simulate_compression_artifacts,
            "moire_pattern": cls.overlay_moire_pattern,
        }
        
        if technique in perturbation_methods:
            return perturbation_methods[technique](image, **kwargs)
        else:
            # Handle checkerboard and arrow separately as they involve PIL Image conversions
            if technique == "checkerboard":
                patch_size = kwargs.get("patch_size", 100)
                square_size = kwargs.get("square_size", 25)
                fill = kwargs.get("fill", 128)
                tiled = kwargs.get("tiled", False)
                
                checker = cls.make_checkerboard(patch_size, square_size, fill)
                return cls.overlay_checkerboard(image, checker, tiled)
            
            elif technique == "random_arrow":
                return cls.add_random_arrow(image)
            
            else:
                raise ValueError(f"Unknown perturbation technique: {technique}")
                
    @staticmethod
    def compute_ssim(original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM) between two images.
        
        Args:
            original: Original image as NumPy array
            perturbed: Perturbed image as NumPy array
            
        Returns:
            SSIM value between 0 (completely different) and 1 (identical)
            
        Note:
            Requires skimage.metrics.structural_similarity to be installed
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            if original.shape != perturbed.shape:
                raise ValueError("Original and perturbed images must have the same shape")
                
            h, w = original.shape[:2]
            win_size = 7
            
            # Adjust window size for small images
            if h < win_size or w < win_size:
                win_size = min(h, w)
                if win_size % 2 == 0:
                    win_size -= 1
                    
            # Handle color images
            if len(original.shape) == 3:
                return ssim(original, perturbed, win_size=win_size, channel_axis=2, data_range=255)
            else:
                return ssim(original, perturbed, win_size=win_size, data_range=255)
                
        except ImportError:
            print("skimage.metrics.structural_similarity is required for SSIM computation")
            return -1.0


def batch_process_images(source_dir: str, 
                         output_dir: str, 
                         techniques: Dict[str, Dict[str, Any]],
                         file_pattern: str = "*.jpg") -> None:
    """
    Process multiple images with different perturbation techniques.
    
    Args:
        source_dir: Directory containing source images
        output_dir: Directory for perturbed images
        techniques: Dictionary mapping technique names to dictionaries of parameters
        file_pattern: File pattern for source images
    """
    import glob
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(source_dir, file_pattern))
    print(f"Found {len(image_files)} images to process")
    
    for i, img_path in enumerate(image_files):
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            # Get filename without directory and extension
            filename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Apply each perturbation technique
            for technique, params in techniques.items():
                # Apply perturbation
                perturbed = ImagePerturbation.perturb_image(image, technique=technique, **params)
                
                # Save perturbed image
                output_path = os.path.join(output_dir, f"{filename}_{technique}.jpg")
                cv2.imwrite(output_path, perturbed)
                
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            
    print("Batch processing complete")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Apply perturbations to medical images")
    parser.add_argument("--source", type=str, required=True, help="Source directory with images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for perturbed images")
    parser.add_argument("--technique", type=str, default="gaussian_noise", 
                        help="Perturbation technique to apply")
    parser.add_argument("--params", type=str, default="{}", 
                        help="JSON string of parameters for the technique")
    
    args = parser.parse_args()
    
    # Import json to parse parameters
    import json
    params = json.loads(args.params)
    
    # Define techniques
    techniques = {args.technique: params}
    
    # Process images
    batch_process_images(args.source, args.output, techniques)