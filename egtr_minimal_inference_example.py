#####################################################################################################################
#
#  To Use This Script:
#  
#  Replace Placeholders: Update the file paths and configuration details at the top (MODEL_CHECKPOINT_PATH, 
#      MODEL_CONFIG_NAME_OR_PATH, OBJECT_LABELS_PATH, REL_LABELS_PATH, INPUT_IMAGE_PATH).
#  
#  Ensure Custom Class: Make absolutely sure the Python file containing the DetrForSceneGraphGeneration class is 
#      accessible (e.g., in the same directory or in your PYTHONPATH). You might need to adjust the import line 
#      (from model.egtr import ...).
#  
#  Install Dependencies: pip install torch torchvision torchaudio transformers opencv-python Pillow
#  
#  Run: python your_minimal_script_name.py
#  
#  This version demonstrates the fundamental flow without the complexities of the previous implementation, making it 
#       easier to understand the core steps involved in using an EGTR model for inference.
#  
#  
#  Assumptions:
#  
#  You have the necessary libraries installed (torch, transformers, opencv-python, Pillow).
#  
#  You have a trained EGTR model checkpoint (.ckpt file).
#  
#  You have the corresponding model configuration (e.g., config.json or the Hugging Face 
#       identifier like SenseTime/deformable-detr).
#  
#  You have simple text files for object and relationship labels (object_labels.txt, relationship_labels.txt), 
#       one label per line.
#  
#  You have the custom DetrForSceneGraphGeneration model class available in your Python environment (this is 
#       crucial and specific to EGTR).
#
#  Required Resources:
#
#  1. Pre-trained EGTR Models:
#     - Visual Genome Dataset: https://drive.google.com/file/d/18phcRxbrEI7HqIuM2OLAPuwAF5k3pUC2/view?usp=drive_link
#     - Open Images V6 Dataset: https://drive.google.com/file/d/1JqWNwf1QvDsTbGFigEXXN_8qv3VF-lGP/view?usp=drive_link
#
#  2. Base Transformer Models:
#     - DeformableDetr: Download via Hugging Face using "SenseTime/deformable-detr" identifier
#     - ResNet-50 Backbone (optional): 
#       Can be generated using the script in the EGTR main repository or downloaded from:
#       https://github.com/naver-ai/egtr/tree/main/models/backbones
#
#  3. Class Labels:
#     - For Visual Genome: The model checkpoint includes object_labels.txt and relationship_labels.txt
#     - For Open Images: The model checkpoint includes object_labels.txt and relationship_labels.txt
#     - If missing, you can extract them from: https://github.com/naver-ai/egtr/tree/main/dataset
#
#  4. Code Dependencies:
#     - EGTR Implementation: https://github.com/naver-ai/egtr
#     - The DetrForSceneGraphGeneration class can be found in the model/egtr.py file in the repository
#     - Clone the repository and ensure the model directory is in your PYTHONPATH
#
#  Project Repository: https://github.com/naver-ai/egtr
#  Paper: https://arxiv.org/abs/2404.02072
#  
#  Important Notes:
#  
#  - Memory Requirements: The model requires at least 4GB GPU memory for inference with default settings
#  - Processing Time: Expect 50-200ms per image on a modern GPU (depends on image size and GPU capability)
#  - Format Compatibility: This script is compatible with EGTR checkpoints from the official repository
#  - When using Visual Genome models, there are 150 object classes and 50 relationship classes
#  - When using Open Images models, there are 601 object classes and 30 relationship classes
#
######################################################################################################################

import os
import cv2
import torch
import numpy as np
from PIL import Image
import logging

# --- Minimal Configuration ---
# --- !! Adjust these paths/values !! ---
MODEL_CHECKPOINT_PATH = "path/to/your/model/checkpoints/epoch=XYZ.ckpt" # Direct path to the checkpoint file
MODEL_CONFIG_NAME_OR_PATH = "SenseTime/deformable-detr" # Or path to a local config.json
OBJECT_LABELS_PATH = "path/to/your/model/object_labels.txt"
REL_LABELS_PATH = "path/to/your/model/relationship_labels.txt"
INPUT_IMAGE_PATH = "path/to/your/image.jpg"
CONFIDENCE_THRESHOLD = 0.5 # Threshold for detecting objects and relationships
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- End Configuration ---

# Configure basic logging for feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import the necessary model components
try:
    from transformers import DeformableDetrFeatureExtractor, DeformableDetrConfig
    # This is the custom EGTR class - YOU MUST HAVE THIS CODE AVAILABLE
    from model.egtr import DetrForSceneGraphGeneration # Replace 'model.egtr' if it's located elsewhere
except ImportError as e:
    logging.error(f"ImportError: {e}")
    logging.error("Ensure 'transformers' is installed and the custom 'DetrForSceneGraphGeneration' class is in your Python path.")
    exit(1)

def load_labels(file_path: str) -> list[str]:
    """Loads labels from a simple text file (one label per line)."""
    try:
        with open(file_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(labels)} labels from {file_path}")
        return labels
    except FileNotFoundError:
        logging.error(f"Label file not found: {file_path}")
        exit(1)

def load_minimal_model(ckpt_path: str, config_name_or_path: str) -> tuple:
    """Loads the minimal model components."""
    logging.info(f"Loading model from checkpoint: {ckpt_path}")
    logging.info(f"Using config: {config_name_or_path}")

    try:
        # 1. Load Feature Extractor & Config
        feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(config_name_or_path)
        config = DeformableDetrConfig.from_pretrained(config_name_or_path)

        # 2. Initialize Model Structure (using the custom EGTR class)
        model = DetrForSceneGraphGeneration.from_pretrained(
            config_name_or_path,
            config=config,
            ignore_mismatched_sizes=True # Important if fine-tuning changed head sizes
        )

        # 3. Load Weights from Checkpoint
        if not os.path.exists(ckpt_path):
             logging.error(f"Checkpoint file not found: {ckpt_path}")
             exit(1)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] # Assuming weights are under 'state_dict'

        # Remove 'model.' prefix if present (common in PyTorch Lightning checkpoints)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict)
        model.to(DEVICE)
        model.eval()
        logging.info("Model loaded successfully.")
        return feature_extractor, model

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        exit(1)

def main():
    # 1. Load Labels
    obj_labels = load_labels(OBJECT_LABELS_PATH)
    rel_labels = load_labels(REL_LABELS_PATH)

    # 2. Load Model
    feature_extractor, model = load_minimal_model(MODEL_CHECKPOINT_PATH, MODEL_CONFIG_NAME_OR_PATH)

    # 3. Load and Preprocess Image
    logging.info(f"Loading image: {INPUT_IMAGE_PATH}")
    if not os.path.exists(INPUT_IMAGE_PATH):
        logging.error(f"Input image not found: {INPUT_IMAGE_PATH}")
        exit(1)

    image_bgr = cv2.imread(INPUT_IMAGE_PATH)
    if image_bgr is None:
        logging.error(f"Failed to read image: {INPUT_IMAGE_PATH}")
        exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    img_height, img_width = image_rgb.shape[:2]

    logging.info("Preprocessing image...")
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    pixel_mask = inputs["pixel_mask"].to(DEVICE) # Often needed by DETR models

    # 4. Inference
    logging.info("Running inference...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    logging.info("Inference complete.")

    # 5. Post-process Outputs
    logging.info("Post-processing results...")
    logits = outputs.logits[0]  # Batch size is 1, take the first element
    boxes = outputs.pred_boxes[0]
    rel_logits = outputs.rel_logits[0] # Relationship logits

    # Get object predictions
    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)

    # Filter based on confidence threshold
    keep = scores > CONFIDENCE_THRESHOLD
    kept_scores = scores[keep].cpu().tolist()
    kept_labels = labels[keep].cpu().tolist()
    kept_boxes = boxes[keep].cpu() # Keep as tensor for now

    # Convert boxes from [center_x, center_y, width, height] (normalized) to [x1, y1, x2, y2] (pixel coords)
    kept_boxes_xyxy = []
    for box in kept_boxes:
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * img_width)
        y1 = int((cy - h / 2) * img_height)
        x2 = int((cx + w / 2) * img_width)
        y2 = int((cy + h / 2) * img_height)
        kept_boxes_xyxy.append([x1, y1, x2, y2])

    logging.info(f"Found {len(kept_labels)} objects passing threshold.")

    # Get relationship predictions (between *kept* objects)
    detected_relations = []
    if len(kept_labels) > 1: # Need at least two objects for a relationship
        keep_indices = torch.where(keep)[0] # Original indices of kept objects
        rel_probs = rel_logits.softmax(-1) # Softmax over relation types

        for i, idx_subj in enumerate(keep_indices):
            for j, idx_obj in enumerate(keep_indices):
                if i == j: continue # Skip self-relations

                # Get the scores for relations between these two specific objects
                rel_scores_for_pair = rel_probs[idx_subj, idx_obj]
                rel_score, rel_label = rel_scores_for_pair.max(-1)

                if rel_score > CONFIDENCE_THRESHOLD:
                    detected_relations.append({
                        "subject_idx": i, # Index within the *kept* objects list
                        "object_idx": j,  # Index within the *kept* objects list
                        "score": rel_score.item(),
                        "label_idx": rel_label.item(),
                    })
    logging.info(f"Found {len(detected_relations)} relationships passing threshold.")

    # 6. Visualize Results (Basic)
    vis_image = image_bgr.copy() # Draw on the BGR image loaded by OpenCV

    # Draw object boxes and labels
    for i, (box, label_idx, score) in enumerate(zip(kept_boxes_xyxy, kept_labels, kept_scores)):
        x1, y1, x2, y2 = box
        label_str = obj_labels[label_idx]
        text = f"{i}: {label_str} ({score:.2f})"

        # Draw rectangle
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label text
        cv2.putText(vis_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print relationships to console
    if detected_relations:
        print("\n--- Detected Relationships ---")
        for rel in detected_relations:
            subj_idx = rel["subject_idx"]
            obj_idx = rel["object_idx"]
            rel_label = rel_labels[rel["label_idx"]]
            score = rel["score"]

            subj_label = obj_labels[kept_labels[subj_idx]]
            obj_label = obj_labels[kept_labels[obj_idx]]

            print(f"{subj_idx}:{subj_label} --[{rel_label} ({score:.2f})]--> {obj_idx}:{obj_label}")
        print("----------------------------\n")

    # Display the image
    logging.info("Displaying results. Press any key to exit.")
    cv2.imshow("EGTR Minimal Output", vis_image)
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

