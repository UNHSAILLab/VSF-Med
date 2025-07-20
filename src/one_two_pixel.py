import os
import ast
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.optimize import differential_evolution
import torchxrayvision as xrv

def create_one_pixel_attacks(df, source_folder, output_folder='deid_png_onepix', 
                           pathology='Pneumonia', max_iter=100, pop_size=200):
    """
    Create one-pixel adversarial attacks for X-ray images
    
    Args:
        df: DataFrame with 'image_path' column
        source_folder: Base folder containing original images
        output_folder: Output folder for adversarial images
        pathology: Target pathology for attack
        max_iter: Maximum iterations for optimization
        pop_size: Population size for differential evolution
    """
    
    # Setup model
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    pathology_idx = model.pathologies.index(pathology)
    
    # Create attacker
    attacker = OnePixelAttackLungConstrained(model, device)
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating one-pixel attacks"):
        try:
            # Parse image paths
            if isinstance(row['image_path'], str):
                image_paths = ast.literal_eval(row['image_path'])
            else:
                image_paths = row['image_path']
            
            for img_path in image_paths:
                # Clean path
                relative_path = img_path.replace('../', '')
                full_input_path = os.path.join(source_folder, relative_path)
                
                # Create output path
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Skip if already exists
                if os.path.exists(output_path):
                    continue
                
                # Load and preprocess image
                img = Image.open(full_input_path)
                
                # Handle different image modes
                if img.mode == 'I;16':
                    img_array = np.array(img)
                    img_min, img_max = img_array.min(), img_array.max()
                    img_8bit = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_8bit = np.array(img.convert('L'))
                
                # Resize to 224x224
                from skimage.transform import resize
                image_224 = resize(img_8bit, (224, 224), anti_aliasing=True)
                image_224 = (image_224 * 255).astype(np.uint8)
                
                # Get lung mask
                lung_mask = get_lung_mask_adaptive(image_224)
                attacker.set_lung_mask(lung_mask)
                
                # Get original prediction
                with torch.no_grad():
                    orig_tensor = attacker.preprocess_for_model(image_224)
                    orig_output = model(orig_tensor)
                    orig_prob = torch.sigmoid(orig_output)[0, pathology_idx].cpu().item()
                
                # Determine target (flip prediction)
                true_label = 1 if orig_prob >= 0.5 else 0
                
                # Run attack
                bounds = [(0, 223), (0, 223), (-255, 255)]
                
                result = differential_evolution(
                    func=lambda p: attacker.fitness_function(p, image_224, true_label, pathology_idx),
                    bounds=bounds,
                    maxiter=max_iter,
                    popsize=pop_size,
                    seed=42,
                    workers=1
                )
                
                # Apply perturbation
                best_x, best_y = int(result.x[0]), int(result.x[1])
                best_delta = result.x[2]
                
                adv_image_224 = attacker.perturb_image(image_224, best_x, best_y, best_delta)
                
                # Resize back to original dimensions
                adv_image_full = resize(adv_image_224, img_8bit.shape, anti_aliasing=True)
                adv_image_full = (adv_image_full * 255).astype(np.uint8)
                
                # Save adversarial image
                Image.fromarray(adv_image_full).save(output_path)
                
                # Log success
                with torch.no_grad():
                    adv_tensor = attacker.preprocess_for_model(adv_image_224)
                    adv_output = model(adv_tensor)
                    adv_prob = torch.sigmoid(adv_output)[0, pathology_idx].cpu().item()
                
                print(f"\n✓ {os.path.basename(output_path)}: {orig_prob:.3f} → {adv_prob:.3f}")
                
        except Exception as e:
            print(f"\n✗ Error processing row {idx}: {e}")
            continue

def create_two_pixel_attacks(df, source_folder, output_folder='deid_png_twopix',
                           pathology='Pneumonia', max_iter=150, pop_size=250):
    """
    Create two-pixel adversarial attacks for X-ray images
    
    Args:
        df: DataFrame with 'image_path' column
        source_folder: Base folder containing original images
        output_folder: Output folder for adversarial images
        pathology: Target pathology for attack
        max_iter: Maximum iterations for optimization
        pop_size: Population size for differential evolution
    """
    
    # Setup model
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    pathology_idx = model.pathologies.index(pathology)
    
    # Create attacker
    attacker = TwoPixelAttackLungConstrained(model, device)
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating two-pixel attacks"):
        try:
            # Parse image paths
            if isinstance(row['image_path'], str):
                image_paths = ast.literal_eval(row['image_path'])
            else:
                image_paths = row['image_path']
            
            for img_path in image_paths:
                # Clean path
                relative_path = img_path.replace('../', '')
                full_input_path = os.path.join(source_folder, relative_path)
                
                # Create output path
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Skip if already exists
                if os.path.exists(output_path):
                    continue
                
                # Load and preprocess image
                img = Image.open(full_input_path)
                
                # Handle different image modes
                if img.mode == 'I;16':
                    img_array = np.array(img)
                    img_min, img_max = img_array.min(), img_array.max()
                    img_8bit = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_8bit = np.array(img.convert('L'))
                
                # Resize to 224x224
                from skimage.transform import resize
                image_224 = resize(img_8bit, (224, 224), anti_aliasing=True)
                image_224 = (image_224 * 255).astype(np.uint8)
                
                # Get lung mask
                lung_mask = get_lung_mask_adaptive(image_224)
                attacker.set_lung_mask(lung_mask)
                
                # Get original prediction
                with torch.no_grad():
                    orig_tensor = attacker.preprocess_for_model(image_224)
                    orig_output = model(orig_tensor)
                    orig_prob = torch.sigmoid(orig_output)[0, pathology_idx].cpu().item()
                
                # Determine target (flip prediction)
                true_label = 1 if orig_prob >= 0.5 else 0
                
                # Run two-pixel attack
                bounds_2px = [
                    (0, 223), (0, 223), (-255, 255),  # pixel 1
                    (0, 223), (0, 223), (-255, 255)   # pixel 2
                ]
                
                result = differential_evolution(
                    func=lambda p: attacker.fitness_function(p, image_224, true_label, pathology_idx),
                    bounds=bounds_2px,
                    maxiter=max_iter,
                    popsize=pop_size,
                    seed=42,
                    workers=1
                )
                
                # Apply perturbations
                x1, y1 = int(result.x[0]), int(result.x[1])
                d1 = result.x[2]
                x2, y2 = int(result.x[3]), int(result.x[4])
                d2 = result.x[5]
                
                adv_image_224 = attacker.perturb_image(image_224, x1, y1, d1, x2, y2, d2)
                
                # Resize back to original dimensions
                adv_image_full = resize(adv_image_224, img_8bit.shape, anti_aliasing=True)
                adv_image_full = (adv_image_full * 255).astype(np.uint8)
                
                # Save adversarial image
                Image.fromarray(adv_image_full).save(output_path)
                
                # Log success
                with torch.no_grad():
                    adv_tensor = attacker.preprocess_for_model(adv_image_224)
                    adv_output = model(adv_tensor)
                    adv_prob = torch.sigmoid(adv_output)[0, pathology_idx].cpu().item()
                
                print(f"\n✓ {os.path.basename(output_path)}: {orig_prob:.3f} → {adv_prob:.3f}")
                print(f"  Pixels: ({x1},{y1}) and ({x2},{y2})")
                
        except Exception as e:
            print(f"\n✗ Error processing row {idx}: {e}")
            continue
