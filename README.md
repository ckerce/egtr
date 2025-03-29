# EGTR Inference Implementation

This repository contains a Python implementation for running inference with the EGTR (Extracting Graph from Transformer) model for scene graph generation (original implementation here https://github.com/naver-ai/egtr). EGTR is a lightweight one-stage scene graph generation model that leverages the multi-head self-attention layers of transformer decoders to extract relationship graphs.

## Overview

EGTR was introduced in the paper ["EGTR: Extracting Graph from Transformer for Scene Graph Generation"](https://arxiv.org/abs/2404.02072) (CVPR 2024 Best Paper Award Candidate). The model efficiently extracts relationships between objects by utilizing the self-attention patterns already present in transformer-based object detectors, eliminating the need for complex relationship modeling.

This implementation provides a simple, inference-only pipeline for generating scene graphs from images using pre-trained EGTR models.

## Features

- Load pre-trained EGTR models
- Process images from files or camera
- Generate scene graphs with objects and their relationships
- Visualize detected objects and relationships
- Export scene graphs to JSON format
- Benchmark inference performance (FPS)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for reasonable performance)

### Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/egtr-inference.git
cd egtr-inference

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

The main dependencies include:
- PyTorch (>=1.10.0)
- torchvision
- transformers
- opencv-python
- Pillow
- matplotlib
- networkx

For full dependency list, see the `requirements.txt` file.

## Downloading Pre-trained Models

Pre-trained EGTR models are available for two datasets:

- **Visual Genome (VG)**: [Download](https://drive.google.com/file/d/18phcRxbrEI7HqIuM2OLAPuwAF5k3pUC2/view?usp=drive_link)
- **Open Images V6 (OI)**: [Download](https://drive.google.com/file/d/1JqWNwf1QvDsTbGFigEXXN_8qv3VF-lGP/view?usp=drive_link)

Download the desired model and extract it to a location of your choice.

## Usage

### Basic Python API

```python
from scene_graph_generator import SceneGraphGenerator

# Initialize the generator with a pre-trained model
generator = SceneGraphGenerator(
    model_path="path/to/model_directory",
    confidence_threshold=0.5
)

# Process an image
image = generator.load_image("path/to/image.jpg")
outputs, _ = generator.generate_scene_graph(image)

# Extract objects and relationships
objects, relationships = generator.process_outputs(outputs)

# Build a graph representation
graph = generator.build_scene_graph(objects, relationships)

# Visualize the results
generator.visualize_scene_graph(image, graph, output_path="scene_graph.png")

# Export to JSON
generator.export_scene_graph(graph, "scene_graph.json")
```

### Command-Line Interface

The implementation includes a command-line interface for easy usage:

```bash
python scene_graph_generator.py --model_path path/to/model_directory --image path/to/image.jpg
```

#### Arguments

- `--model_path`: Path to the directory containing the trained EGTR model (required)
- `--image`: Path to input image (if not provided, will try to capture from camera)
- `--camera`: Camera index to use (default: 0)
- `--output`: Path to save visualization output (optional)
- `--json_output`: Path to save JSON output (optional)
- `--threshold`: Confidence threshold for predictions (default: 0.5)
- `--min_size`: Minimum image size for preprocessing (default: 800)
- `--max_size`: Maximum image size for preprocessing (default: 1333)
- `--device`: Device to use ('cuda' or 'cpu')
- `--benchmark`: Run FPS benchmark
- `--no_vis`: Skip visualization

## Model Architecture

EGTR is built on top of transformer-based object detectors (specifically Deformable DETR) and makes the key innovation of extracting relationship graphs directly from the multi-head self-attention layers. This approach offers several advantages:

1. **Efficiency**: Avoids complex relationship modeling
2. **Lightweight**: Uses a shallow relation extraction head
3. **Performance**: Achieves competitive results with less computational overhead

The model employs novel techniques such as relation smoothing and connectivity prediction to improve performance.

## Output Format

### Objects

Each detected object contains:
- `id`: Object identifier
- `label`: Object class label
- `label_idx`: Object class index
- `box`: Bounding box coordinates [x1, y1, x2, y2]
- `score`: Detection confidence score

### Relationships

Each relationship contains:
- `subject_idx`: Index of the subject object
- `object_idx`: Index of the object 
- `relation`: Relationship class label
- `relation_idx`: Relationship class index
- `score`: Relationship confidence score

## Examples

### Visualization Output

The visualization includes:
1. The original image with detected objects (bounding boxes)
2. A graph representation of the scene with objects as nodes and relationships as edges

### JSON Output Example

```json
{
  "objects": [
    {
      "id": 0,
      "label": "person",
      "label_idx": 1,
      "score": 0.98,
      "x1": 100,
      "y1": 150,
      "x2": 200,
      "y2": 350
    },
    {
      "id": 1,
      "label": "bicycle",
      "label_idx": 2,
      "score": 0.89,
      "x1": 250,
      "y1": 200,
      "x2": 400,
      "y2": 350
    }
  ],
  "relationships": [
    {
      "label": "riding",
      "score": 0.86,
      "relation_idx": 15,
      "subject_idx": 0,
      "object_idx": 1
    }
  ]
}
```

## Performance Benchmarking

You can benchmark the performance of the EGTR model on your hardware:

```bash
python scene_graph_generator.py --model_path path/to/model_directory --image path/to/image.jpg --benchmark
```

Typical performance on a modern GPU:
- NVIDIA V100: ~15-20 FPS
- NVIDIA RTX 3080: ~12-18 FPS
- NVIDIA GTX 1080 Ti: ~8-12 FPS

## Troubleshooting

### Common Issues

1. **Model loading fails**: 
   - Ensure the model directory contains checkpoint files
   - Check that your PyTorch version is compatible with the saved checkpoint

2. **Out of memory errors**:
   - Try reducing `min_size` and `max_size` for image preprocessing
   - Process images one at a time

3. **Relationship prediction issues**:
   - Verify that the model was trained with relationship prediction head
   - If no relationships are detected, try lowering the confidence threshold

### Environment Setup

For optimal performance, we recommend using the same environment used for model training:
- Docker image: `nvcr.io/nvidia/pytorch:21.11-py3`

## Citation

If you use EGTR in your research, please cite the original paper:

```
@InProceedings{Im_2024_CVPR,
    author    = {Im, Jinbae and Nam, JeongYeon and Park, Nokyung and Lee, Hyungmin and Park, Seunghyun},
    title     = {EGTR: Extracting Graph from Transformer for Scene Graph Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {24229-24238}
}
```

## License

This implementation is provided under the Apache License 2.0.

## Acknowledgments

- This implementation is based on the official [EGTR repository](https://github.com/naver-ai/egtr)
- Transformer components rely on HuggingFace Transformers library
