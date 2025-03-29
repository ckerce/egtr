import os
import cv2
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from typing import Dict, List, Tuple, Optional, Union
import logging
from glob import glob
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import transformer model components - these would need to be installed
# Using try/except to make the implementation more robust
try:
    from transformers import DeformableDetrFeatureExtractor, DeformableDetrConfig
    from model.egtr import DetrForSceneGraphGeneration
except ImportError:
    logger.warning("Transformer models not found. Please install the required packages.")
    logger.warning("If using HuggingFace Transformers, run: pip install transformers")
    logger.warning("If using custom EGTR implementation, ensure it's in your Python path")


class SceneGraphGenerator:
    """
    Enhanced Scene Graph Generator using EGTR model.
    
    This class implements an inference-only scene graph generator that takes an image as input
    and produces a scene graph representing objects and their relationships.
    """
    
    def __init__(
        self, 
        model_path: str,
        architecture: str = "SenseTime/deformable-detr",
        min_size: int = 800,
        max_size: int = 1333,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the Scene Graph Generator.
        
        Args:
            model_path (str): Path to the trained EGTR model
            architecture (str): Backbone architecture
            min_size (int): Minimum image size for processing
            max_size (int): Maximum image size for processing
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
            confidence_threshold (float): Default confidence threshold for predictions
        """
        # Set up device
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Set confidence threshold
        self.confidence_threshold = confidence_threshold
        
        # Store image preprocessing parameters
        self.min_size = min_size
        self.max_size = max_size
        
        # Load the model
        self.feature_extractor, self.model = self._load_model(model_path, architecture)
        
        # Load labels
        self.obj_labels = self._load_object_labels(model_path)
        self.rel_labels = self._load_relationship_labels(model_path)
        
        logger.info(f"Scene Graph Generator initialized with {len(self.obj_labels)} object classes and {len(self.rel_labels)} relationship classes")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """
        Set up the device for model inference.
        
        Args:
            device (str, optional): Device specification ('cuda' or 'cpu')
            
        Returns:
            torch.device: The device to use
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return torch.device(device)
    
    def _find_latest_checkpoint(self, model_path: str) -> str:
        """
        Find the latest checkpoint in the model directory.
        
        Args:
            model_path (str): Path to the model directory
            
        Returns:
            str: Path to the latest checkpoint
        """
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory {checkpoint_dir} not found")
            # Check if the model_path itself is a checkpoint file
            if os.path.isfile(model_path) and model_path.endswith('.ckpt'):
                logger.info(f"Using provided checkpoint file: {model_path}")
                return model_path
            else:
                raise FileNotFoundError(f"No checkpoint found at {model_path}")
        
        # Find all checkpoint files
        checkpoint_files = glob(os.path.join(checkpoint_dir, "epoch=*.ckpt"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Sort by epoch number
        latest_checkpoint = sorted(
            checkpoint_files,
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
        
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    def _load_model(
        self, 
        model_path: str,
        architecture: str,
    ) -> Tuple[DeformableDetrFeatureExtractor, DetrForSceneGraphGeneration]:
        """
        Load the EGTR model and feature extractor.
        
        Args:
            model_path (str): Path to the model directory
            architecture (str): Backbone architecture name
            
        Returns:
            Tuple[DeformableDetrFeatureExtractor, DetrForSceneGraphGeneration]: 
                Feature extractor and model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Initialize feature extractor
        try:
            feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
                architecture, 
                size=self.min_size,
                max_size=self.max_size
            )
        except Exception as e:
            logger.error(f"Error loading feature extractor: {e}")
            raise
        
        # Load model configuration
        try:
            if os.path.exists(os.path.join(model_path, "config.json")):
                config = DeformableDetrConfig.from_pretrained(model_path)
            else:
                # If config not found in model_path, try loading from architecture
                config = DeformableDetrConfig.from_pretrained(architecture)
                logger.warning(f"Config not found in {model_path}, using default from {architecture}")
        except Exception as e:
            logger.error(f"Error loading model configuration: {e}")
            raise
        
        # Initialize model
        try:
            model = DetrForSceneGraphGeneration.from_pretrained(
                architecture, 
                config=config,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
        
        # Load model weights
        try:
            ckpt_path = self._find_latest_checkpoint(model_path)
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            
            # Handle prefix in state dict (model.* -> *)
            for k in list(state_dict.keys()):
                if k.startswith("model."):
                    state_dict[k[6:]] = state_dict.pop(k)
            
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise
        
        # Move model to device and set to evaluation mode
        model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return feature_extractor, model
    
    def _load_object_labels(self, model_path: str) -> List[str]:
        """
        Load object class labels.
        
        Args:
            model_path (str): Path to the model directory
            
        Returns:
            List[str]: List of object class labels
        """
        # Try to load from a file in the model path
        labels_path = os.path.join(model_path, "object_labels.txt")
        
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Try to load from JSON file
        labels_json_path = os.path.join(model_path, "object_labels.json")
        if os.path.exists(labels_json_path):
            with open(labels_json_path, 'r') as f:
                return json.load(f)
        
        # If files don't exist, return placeholder labels and warn
        logger.warning(f"Object labels file not found at {labels_path} or {labels_json_path}")
        logger.warning("Using placeholder object labels")
        
        # Return placeholder labels (150 is typical for Visual Genome)
        return [f"object_{i}" for i in range(150)]
    
    def _load_relationship_labels(self, model_path: str) -> List[str]:
        """
        Load relationship class labels.
        
        Args:
            model_path (str): Path to the model directory
            
        Returns:
            List[str]: List of relationship class labels
        """
        # Try to load from a text file in the model path
        labels_path = os.path.join(model_path, "relationship_labels.txt")
        
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Try to load from JSON file
        labels_json_path = os.path.join(model_path, "relationship_labels.json")
        if os.path.exists(labels_json_path):
            with open(labels_json_path, 'r') as f:
                return json.load(f)
        
        # If files don't exist, return placeholder labels and warn
        logger.warning(f"Relationship labels file not found at {labels_path} or {labels_json_path}")
        logger.warning("Using placeholder relationship labels")
        
        # Return placeholder labels (50 is typical for Visual Genome)
        return [f"relation_{i}" for i in range(50)]
    
    def capture_image(self, camera_index: int = 0) -> np.ndarray:
        """
        Capture an image from the camera.
        
        Args:
            camera_index (int): Index of the camera to use
            
        Returns:
            np.ndarray: The captured image
        """
        cap = cv2.VideoCapture(camera_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Failed to capture image from camera {camera_index}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.info(f"Captured image from camera {camera_index} with shape {image.shape}")
        
        return image
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from a file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: The loaded image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Loaded image from {image_path} with shape {image.shape}")
        
        return image
    
    def generate_scene_graph(self, image: np.ndarray) -> Tuple[Dict, torch.Tensor]:
        """
        Generate a scene graph from an image.
        
        Args:
            image (np.ndarray): Input image in RGB format
            
        Returns:
            Tuple[Dict, torch.Tensor]: Model outputs and processed image tensor
        """
        start_time = time.time()
        
        # Convert numpy image to PIL
        pil_image = Image.fromarray(image)
        
        # Process image through feature extractor
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        
        # Move tensors to device
        pixel_values = inputs["pixel_values"].to(self.device)
        pixel_mask = inputs["pixel_mask"].to(self.device)
        
        # Generate scene graph
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                output_attentions=False,
                output_attention_states=True,
                output_hidden_states=True,
            )
        
        end_time = time.time()
        logger.info(f"Scene graph generation completed in {end_time - start_time:.3f} seconds")
        
        return outputs, pixel_values
    
    def process_outputs(
        self, 
        outputs: Dict, 
        confidence_threshold: Optional[float] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process model outputs to extract objects and relationships.
        
        Args:
            outputs (Dict): Model outputs
            confidence_threshold (float, optional): Confidence threshold for predictions
            
        Returns:
            Tuple[List[Dict], List[Dict]]: Lists of objects and relationships
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Extract object predictions
        logits = outputs.logits[0]  # Shape: [num_queries, num_classes]
        boxes = outputs.pred_boxes[0]  # Shape: [num_queries, 4]
        
        # Get object class predictions
        probs = torch.nn.functional.softmax(logits, dim=-1)
        scores, labels = probs.max(-1)
        
        # Filter predictions by confidence
        keep = scores > confidence_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
        
        # Extract relationship predictions
        relationships = []
        obj_indices = torch.where(keep)[0]
        
        if hasattr(outputs, 'rel_logits'):
            rel_logits = outputs.rel_logits[0]  # Shape: [num_queries, num_queries, num_rel_classes]
            
            # Process relationship predictions
            rel_probs = torch.nn.functional.softmax(rel_logits, dim=-1)
            
            # Get the indices of the kept objects
            kept_indices = torch.where(keep)[0]
            
            # Extract relationships only between kept objects
            for i, src_idx in enumerate(kept_indices):
                for j, dst_idx in enumerate(kept_indices):
                    if i != j:  # Avoid self-relationships
                        # Get relationship scores and labels for this pair
                        rel_scores = rel_probs[src_idx, dst_idx]
                        rel_label = torch.argmax(rel_scores).item()
                        rel_score = rel_scores[rel_label].item()
                        
                        if rel_score > confidence_threshold:
                            relationships.append({
                                'subject_idx': i,
                                'object_idx': j,
                                'relation': self.rel_labels[rel_label],
                                'relation_idx': rel_label,
                                'score': rel_score
                            })
        else:
            logger.warning("No relationship predictions found in model outputs")
        
        # Create list of objects
        objects = []
        for i, (box, label, score) in enumerate(zip(boxes_xyxy, labels, scores)):
            objects.append({
                'id': i,
                'label': self.obj_labels[label.item()],
                'label_idx': label.item(),
                'box': box.tolist(),
                'score': score.item()
            })
        
        logger.info(f"Processed outputs: {len(objects)} objects and {len(relationships)} relationships")
        return objects, relationships
    
    def build_scene_graph(
        self, 
        objects: List[Dict], 
        relationships: List[Dict]
    ) -> nx.DiGraph:
        """
        Build a NetworkX graph from objects and relationships.
        
        Args:
            objects (List[Dict]): List of object dictionaries
            relationships (List[Dict]): List of relationship dictionaries
            
        Returns:
            nx.DiGraph: The scene graph
        """
        graph = nx.DiGraph()
        
        # Add nodes (objects)
        for obj in objects:
            graph.add_node(obj['id'], **obj)
        
        # Add edges (relationships)
        for rel in relationships:
            graph.add_edge(
                rel['subject_idx'], 
                rel['object_idx'], 
                label=rel['relation'],
                score=rel['score'],
                relation_idx=rel['relation_idx'] if 'relation_idx' in rel else None
            )
        
        logger.info(f"Built scene graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def visualize_scene_graph(
        self, 
        image: np.ndarray, 
        graph: nx.DiGraph, 
        output_path: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ) -> np.ndarray:
        """
        Visualize the scene graph over the image.
        
        Args:
            image (np.ndarray): The original image
            graph (nx.DiGraph): The scene graph
            output_path (str, optional): Path to save the visualization
            show_plot (bool): Whether to display the plot
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            np.ndarray: The visualization image
        """
        # Create a copy of the image for drawing
        vis_image = image.copy()
        
        # Draw bounding boxes and labels
        for node_id, node_data in graph.nodes(data=True):
            if 'box' in node_data:
                box = node_data['box']
                label = node_data['label']
                score = node_data['score']
                
                # Convert to integers for drawing
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label}: {score:.2f}"
                cv2.putText(vis_image, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Create a separate figure for the graph visualization
        plt.figure(figsize=figsize)
        
        # Display the image
        plt.subplot(2, 1, 1)
        plt.imshow(vis_image)
        plt.title("Objects Detected")
        plt.axis('off')
        
        # Display the graph
        plt.subplot(2, 1, 2)
        pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
        
        # Draw node labels
        node_labels = {node: data['label'] for node, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=node_labels)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, width=2, arrowsize=20)
        
        # Draw edge labels
        edge_labels = {(u, v): data['label'] for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        
        plt.title("Scene Graph")
        plt.axis('off')
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        
        return vis_image
    
    def export_scene_graph(
        self, 
        graph: nx.DiGraph, 
        output_path: str,
        include_image: bool = False,
        image_path: Optional[str] = None
    ) -> None:
        """
        Export the scene graph to a JSON file.
        
        Args:
            graph (nx.DiGraph): The scene graph
            output_path (str): Path to save the JSON file
            include_image (bool): Whether to include image path in the JSON
            image_path (str, optional): Path to the original image
        """
        # Create a serializable representation of the graph
        data = {
            "objects": [],
            "relationships": []
        }
        
        if include_image and image_path:
            data["image_path"] = image_path
        
        # Add objects
        for node, attrs in graph.nodes(data=True):
            # Remove the raw box data to make it more readable
            obj_data = {k: v for k, v in attrs.items() if k != 'box'}
            if 'box' in attrs:
                box = attrs['box']
                obj_data['x1'] = box[0]
                obj_data['y1'] = box[1]
                obj_data['x2'] = box[2]
                obj_data['y2'] = box[3]
            
            data["objects"].append(obj_data)
        
        # Add relationships
        for src, dst, attrs in graph.edges(data=True):
            rel_data = attrs.copy()
            rel_data["subject_idx"] = src
            rel_data["object_idx"] = dst
            data["relationships"].append(rel_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Scene graph exported to {output_path}")
    
    def benchmark_fps(
        self, 
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        num_runs: int = 10
    ) -> float:
        """
        Benchmark the frames per second (FPS) for scene graph generation.
        
        Args:
            image (np.ndarray, optional): Input image
            image_path (str, optional): Path to an image file
            num_runs (int): Number of runs to average over
            
        Returns:
            float: Average FPS
        """
        if image is None and image_path is None:
            raise ValueError("Either image or image_path must be provided")
        
        if image is None:
            image = self.load_image(image_path)
        
        # Convert to PIL once to avoid doing it in the loop
        pil_image = Image.fromarray(image)
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        pixel_mask = inputs["pixel_mask"].to(self.device)
        
        # Warm up
        with torch.no_grad():
            self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                output_attentions=False,
                output_attention_states=False,
                output_hidden_states=False,
            )
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    output_attentions=False,
                    output_attention_states=False,
                    output_hidden_states=False,
                )
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        fps = num_runs / elapsed_time
        
        logger.info(f"Average FPS: {fps:.2f} ({elapsed_time/num_runs:.3f} seconds per image)")
        return fps


def main():
    """Main function to run the scene graph generator."""
    parser = argparse.ArgumentParser(description="Enhanced Scene Graph Generator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained EGTR model")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization")
    parser.add_argument("--json_output", type=str, default=None, help="Output path for JSON export")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--min_size", type=int, default=800, help="Minimum image size")
    parser.add_argument("--max_size", type=int, default=1333, help="Maximum image size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--benchmark", action="store_true", help="Run FPS benchmark")
    parser.add_argument("--no_vis", action="store_true", help="Skip visualization")
    args = parser.parse_args()
    
    # Initialize scene graph generator
    generator = SceneGraphGenerator(
        model_path=args.model_path,
        min_size=args.min_size,
        max_size=args.max_size,
        device=args.device,
        confidence_threshold=args.threshold
    )
    
    # Get input image
    if args.image:
        image = generator.load_image(args.image)
    else:
        image = generator.capture_image(args.camera)
    
    # Run benchmark if requested
    if args.benchmark:
        fps = generator.benchmark_fps(image, num_runs=20)
        print(f"Benchmark results: {fps:.2f} FPS")
        return
    
    # Generate scene graph
    outputs, processed_image = generator.generate_scene_graph(image)
    
    # Process outputs
    objects, relationships = generator.process_outputs(outputs, args.threshold)
    
    # Build scene graph
    graph = generator.build_scene_graph(objects, relationships)
    
    # Visualize scene graph
    if not args.no_vis:
        generator.visualize_scene_graph(image, graph, args.output)
    
    # Export scene graph to JSON if requested
    if args.json_output:
        generator.export_scene_graph(
            graph, 
            args.json_output, 
            include_image=args.image is not None,
            image_path=args.image
        )
    
    # Print summary
    print(f"Detected {len(objects)} objects and {len(relationships)} relationships")
    for obj in objects:
        print(f"Object {obj['id']}: {obj['label']} (score: {obj['score']:.2f})")
    
    if relationships:
        print("\nRelationships:")
        for rel in relationships:
            subj = objects[rel['subject_idx']]['label']
            obj = objects[rel['object_idx']]['label']
            print(f"{subj} --[{rel['relation']}]--> {obj} (score: {rel['score']:.2f})")


if __name__ == "__main__":
    main()

