# EGTR Training Guide: Adding New Object Classes and Relationships

This guide provides detailed instructions for training or fine-tuning the EGTR (Extracting Graph from Transformer) model to support new object classes or relationships. The process is broken down into modular components to help you understand which parts need to be modified for specific use cases.

## Table of Contents

1. [Overview of EGTR Architecture](#overview-of-egtr-architecture)
2. [Understanding the Training Pipeline](#understanding-the-training-pipeline)
3. [Preparing Your Dataset](#preparing-your-dataset)
4. [Module 1: Object Detection Training](#module-1-object-detection-training)
5. [Module 2: Relationship Extraction Head Training](#module-2-relationship-extraction-head-training)
6. [Module 3: Joint Fine-tuning](#module-3-joint-fine-tuning)
7. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
8. [Common Issues and Solutions](#common-issues-and-solutions)
9. [Advanced Customization](#advanced-customization)

## Overview of EGTR Architecture

EGTR consists of several key components:

1. **Backbone**: A ResNet-based convolutional network that extracts features from input images
2. **Transformer Encoder**: Processes features from the backbone
3. **Transformer Decoder**: Generates object queries and attends to relevant parts of the image
4. **Object Detection Head**: Predicts object classes and bounding boxes
5. **Relation Extraction Head**: A lightweight component that extracts relationships from self-attention patterns

The model's key innovation is leveraging the multi-head self-attention layers of the transformer decoder to extract relationships between objects, rather than using complex dedicated relationship modeling modules.

## Understanding the Training Pipeline

Training EGTR involves a two-stage process:

1. **Pre-training the object detector**: First, train the object detection components (backbone, transformer encoder/decoder, object detection head) on your dataset
2. **Training the relation extraction head**: Then, train the relation extraction head while fine-tuning the entire model

This approach allows the model to focus on object detection first, then learn to extract relationships once it has a solid understanding of objects.

## Preparing Your Dataset

Before training, you need to prepare your dataset in the correct format:

### Dataset Structure

For training EGTR, your dataset should have:

1. **Images**: Raw image files
2. **Annotations**: JSON files containing:
   - Object instances (classes, bounding boxes)
   - Relationships between objects (subject, predicate, object)

### Data Format

The expected JSON format is:

```json
{
  "images": [
    {
      "file_name": "image1.jpg",
      "height": 800,
      "width": 1200,
      "id": 1
    },
    ...
  ],
  "annotations": [
    {
      "image_id": 1,
      "bbox": [x, y, width, height],
      "category_id": 5,
      "id": 101
    },
    ...
  ],
  "relationships": [
    {
      "subject_id": 101,
      "object_id": 102,
      "predicate_id": 3
    },
    ...
  ],
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
    ...
  ],
  "predicates": [
    {"id": 1, "name": "on"},
    {"id": 2, "name": "has"},
    ...
  ]
}
```

### Data Preprocessing

1. Install preprocessing dependencies:
   ```bash
   pip install -r preprocessing_requirements.txt
   ```

2. Convert your custom dataset to the required format:
   ```bash
   python tools/prepare_dataset.py \
       --input_dir path/to/your/dataset \
       --output_dir dataset/custom_dataset \
       --split train,val,test
   ```

3. Verify the dataset structure:
   ```bash
   python tools/verify_dataset.py --data_path dataset/custom_dataset
   ```

## Module 1: Object Detection Training

The first step is to train or fine-tune the object detection components on your dataset.

### Configuring the Object Detection Model

1. Create a configuration file for your custom dataset:
   ```bash
   cp configs/detr_r50_vg.py configs/detr_r50_custom.py
   ```

2. Edit the configuration file to match your dataset:
   ```python
   # configs/detr_r50_custom.py
   _base_ = 'detr_r50.py'
   
   # Dataset settings
   dataset_type = 'CustomDataset'
   data_root = 'dataset/custom_dataset/'
   
   # Class settings
   num_classes = 150  # Update this to match your number of object classes
   
   # Update other parameters as needed
   ```

### Training the Object Detector

1. Download the ResNet-50 backbone separately:
   ```python
   import torch
   from transformers import PretrainedConfig
   from transformers.models.detr.modeling_detr import DetrTimmConvEncoder, DetrConvModel, build_position_encoding
   
   config = PretrainedConfig.from_pretrained("facebook/detr-resnet-50")
   backbone = DetrTimmConvEncoder(config.backbone, config.dilation)
   position_embeddings = build_position_encoding(config)
   backbone = DetrConvModel(backbone, position_embeddings)
   torch.save(backbone.state_dict(), "models/backbones/resnet50.pt")
   ```

2. Start pre-training the object detector:
   ```bash
   python pretrain_detr.py \
       --data_path dataset/custom_dataset \
       --output_path models/pretrained_detector \
       --backbone_dirpath models/backbones \
       --batch_size 16 \
       --num_classes 150 \
       --epochs 150 \
       --lr 1e-4 \
       --dropout 0.1 \
       --weight_decay 1e-4 \
       --memo "custom_dataset_pretrain"
   ```

3. Monitor training progress:
   ```bash
   tensorboard --logdir models/pretrained_detector/logs
   ```

### Adding New Object Classes

To add new object classes to an existing model:

1. Update your dataset with the new classes
2. Modify the configuration file to include the new classes:
   ```python
   # configs/detr_r50_custom.py
   num_classes = 165  # Updated number of classes
   ```

3. Initialize the classifier weights for new classes:
   ```python
   python tools/expand_classifier.py \
       --pretrained_model path/to/pretrained/model \
       --output_model path/to/output/model \
       --old_num_classes 150 \
       --new_num_classes 165
   ```

4. Fine-tune the model on your updated dataset:
   ```bash
   python pretrain_detr.py \
       --data_path dataset/custom_dataset \
       --output_path models/expanded_detector \
       --pretrained models/pretrained_detector/checkpoints/epoch=xxx.ckpt \
       --num_classes 165 \
       --epochs 50 \
       --lr 5e-5 \
       --memo "expanded_classes_finetune"
   ```

## Module 2: Relationship Extraction Head Training

After training the object detector, you can train the relationship extraction head.

### Configuring the Relationship Extraction Head

1. Create a configuration file for the relation extraction head:
   ```bash
   cp configs/egtr_r50_vg.py configs/egtr_r50_custom.py
   ```

2. Edit the configuration file:
   ```python
   # configs/egtr_r50_custom.py
   _base_ = 'egtr_r50.py'
   
   # Dataset settings
   dataset_type = 'CustomDataset'
   data_root = 'dataset/custom_dataset/'
   
   # Class settings
   num_classes = 165  # Your updated number of object classes
   num_rel_classes = 50  # Number of relationship classes
   
   # Relation extraction settings
   relation_head = dict(
       type='RelationExtractionHead',
       in_channels=256,
       hidden_dim=256,
       num_rel_classes=50,
       dropout=0.1
   )
   ```

### Training the Relationship Extraction Head

1. Start training the full EGTR model, including the relation extraction head:
   ```bash
   python train_egtr.py \
       --data_path dataset/custom_dataset \
       --output_path models/egtr_custom \
       --pretrained models/expanded_detector/checkpoints/epoch=xxx.ckpt \
       --rel_classes 50 \
       --epochs 100 \
       --lr 1e-4 \
       --relation_dropout 0.1 \
       --memo "egtr_custom_dataset"
   ```

### Adding New Relationship Classes

To add new relationship classes to an existing model:

1. Update your dataset with the new relationship classes
2. Modify the configuration file:
   ```python
   # configs/egtr_r50_custom.py
   num_rel_classes = 60  # Updated number of relationship classes
   ```

3. Initialize the relation classifier weights for new classes:
   ```python
   python tools/expand_rel_classifier.py \
       --pretrained_model path/to/pretrained/egtr \
       --output_model path/to/output/model \
       --old_num_rel_classes 50 \
       --new_num_rel_classes 60
   ```

4. Fine-tune the model on your updated dataset:
   ```bash
   python train_egtr.py \
       --data_path dataset/custom_dataset \
       --output_path models/egtr_expanded \
       --pretrained models/egtr_custom/checkpoints/epoch=xxx.ckpt \
       --rel_classes 60 \
       --epochs 50 \
       --lr 5e-5 \
       --memo "expanded_relationships_finetune"
   ```

## Module 3: Joint Fine-tuning

For best results, jointly fine-tune all components after adding new classes or relationships.

### Joint Fine-tuning Procedure

1. Configure joint fine-tuning:
   ```python
   # configs/joint_finetune.py
   _base_ = 'egtr_r50_custom.py'
   
   # Fine-tuning settings
   optimizer = dict(
       type='AdamW',
       lr=5e-5,
       weight_decay=1e-4,
   )
   lr_config = dict(
       policy='step',
       warmup='linear',
       warmup_iters=500,
       step=[30, 45]
   )
   ```

2. Start joint fine-tuning:
   ```bash
   python train_egtr.py \
       --config configs/joint_finetune.py \
       --data_path dataset/custom_dataset \
       --output_path models/egtr_joint_finetune \
       --pretrained models/egtr_expanded/checkpoints/epoch=xxx.ckpt \
       --epochs 50 \
       --lr 5e-5 \
       --joint_finetune True \
       --memo "joint_finetune_custom_dataset"
   ```

## Evaluation and Benchmarking

### Evaluating Your Trained Model

1. Evaluate object detection performance:
   ```bash
   python evaluate_egtr.py \
       --data_path dataset/custom_dataset \
       --artifact_path models/egtr_joint_finetune \
       --eval_mode detection \
       --batch_size 16
   ```

2. Evaluate scene graph generation performance:
   ```bash
   python evaluate_egtr.py \
       --data_path dataset/custom_dataset \
       --artifact_path models/egtr_joint_finetune \
       --eval_mode sggen \
       --batch_size 16
   ```

### Performance Metrics

The evaluation will output various metrics:

- **Object Detection**: mAP@0.5, mAP@0.5:0.95
- **Scene Graph Generation**: 
  - Recall@K (R@50, R@100)
  - Mean Recall@K (mR@50, mR@100)
  - Recall@K per relationship class

### Benchmarking Inference Speed

Benchmark your model's inference speed:

```bash
python evaluate_egtr.py \
    --data_path dataset/custom_dataset \
    --artifact_path models/egtr_joint_finetune \
    --min_size 600 \
    --max_size 1000 \
    --infer_only True
```

## Common Issues and Solutions

### Class Imbalance

**Problem**: Imbalanced distribution of object or relationship classes leads to poor performance on rare classes.

**Solution**:
- Implement class balancing in the loss function:
  ```python
  # In configs/egtr_r50_custom.py
  loss_cls = dict(
      type='FocalLoss',
      use_sigmoid=True,
      gamma=2.0,
      alpha=0.25,
      loss_weight=2.0
  )
  ```

- Apply relationship smoothing (a technique introduced in EGTR):
  ```python
  # In configs/egtr_r50_custom.py
  relation_head = dict(
      type='RelationExtractionHead',
      relation_smoothing=True,
      smoothing_weight=0.1
  )
  ```

### Overfitting

**Problem**: Model performs well on training data but poorly on validation data.

**Solution**:
- Increase dropout rate:
  ```python
  # In configs/egtr_r50_custom.py
  dropout = 0.2  # Increase from default 0.1
  relation_dropout = 0.3  # Increase from default 0.1
  ```

- Apply regularization techniques:
  ```python
  # In configs/egtr_r50_custom.py
  weight_decay = 1e-3  # Increase from default 1e-4
  ```

### Limited Data

**Problem**: Not enough data for some classes or relationships.

**Solution**:
- Implement data augmentation:
  ```python
  # In configs/egtr_r50_custom.py
  train_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(type='LoadAnnotations', with_bbox=True, with_rel=True),
      dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
      dict(type='RandomFlip', flip_ratio=0.5),
      dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2),
      dict(type='Normalize', **img_norm_cfg),
      dict(type='Pad', size_divisor=32),
      dict(type='DefaultFormatBundle'),
      dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels']),
  ]
  ```

- Use transfer learning from a similar domain:
  ```bash
  python train_egtr.py \
      --data_path dataset/custom_dataset \
      --output_path models/egtr_transfer \
      --pretrained models/egtr_visual_genome/checkpoints/epoch=xxx.ckpt \
      --transfer_learning True \
      --freeze_backbone False \
      --epochs 100
  ```

## Advanced Customization

### Custom Transformer Architecture

For specialized applications, you may want to modify the transformer architecture:

1. Create a custom transformer configuration:
   ```python
   # configs/custom_transformer.py
   
   # Custom encoder configuration
   encoder = dict(
       type='DetrTransformerEncoder',
       num_layers=8,  # Increased from 6
       layer_cfg=dict(
           self_attn_cfg=dict(
               embed_dims=256,
               num_heads=8,
               dropout=0.1
           ),
           ffn_cfg=dict(
               embed_dims=256,
               feedforward_channels=2048,
               ffn_drop=0.1
           )
       )
   )
   
   # Custom decoder configuration
   decoder = dict(
       type='DetrTransformerDecoder',
       num_layers=8,  # Increased from 6
       layer_cfg=dict(
           self_attn_cfg=dict(
               embed_dims=256,
               num_heads=8,
               dropout=0.1
           ),
           cross_attn_cfg=dict(
               embed_dims=256,
               num_heads=8,
               dropout=0.1
           ),
           ffn_cfg=dict(
               embed_dims=256,
               feedforward_channels=2048,
               ffn_drop=0.1
           )
       )
   )
   ```

2. Reference your custom transformer in the model configuration:
   ```python
   # configs/egtr_custom.py
   transformer = dict(
       encoder=encoder,  # From custom_transformer.py
       decoder=decoder,  # From custom_transformer.py
   )
   ```

### Custom Data Loading and Preprocessing

For special data requirements:

1. Create a custom dataset class:
   ```python
   # datasets/custom_dataset.py
   from .base_dataset import BaseDataset
   
   class CustomDataset(BaseDataset):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Custom initialization
       
       def load_annotations(self, ann_file):
           # Custom annotation loading
           
       def prepare_training_img(self, idx):
           # Custom image preparation for training
   ```

2. Register your custom dataset:
   ```python
   # datasets/__init__.py
   from .custom_dataset import CustomDataset
   
   __all__ = ['CustomDataset']
   ```

3. Reference your custom dataset in the configuration:
   ```python
   # configs/egtr_custom.py
   dataset_type = 'CustomDataset'
   ```

### Multi-GPU Training

For faster training on multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_egtr.py \
    --data_path dataset/custom_dataset \
    --output_path models/egtr_multigpu \
    --pretrained models/pretrained_detector/checkpoints/epoch=xxx.ckpt \
    --distributed True \
    --batch_size 4 \  # Per GPU
    --epochs 100
```

## Final Notes

- Always back up your model checkpoints before making significant changes
- Regularly evaluate on a validation set to monitor progress
- Start with a small learning rate when fine-tuning pre-trained models
- Consider using learning rate schedulers for better convergence
- Experiment with different hyperparameters to find the optimal configuration for your specific use case

By following this guide, you should be able to successfully add new object classes or relationships to the EGTR model, or adapt it to entirely new domains.
