"""
ADAPT-Inspired Neural Model for Driving Explanation Classification.

This module implements the multimodal classification model described in the paper,
featuring:
- Temporal video tokens (T segments)
- FiLM-based action conditioning
- Asymmetric cross-attention (Video ← Action)
- Multi-prototype classification head with log-sum-exp
- AM-Softmax (Angular Margin) loss

Usage:
    # Training
    python neural_model/train.py \
        --action-cache data/baseline/action_features_sbert.npz \
        --video-cache data/baseline/video_features_resnet18_mean.npz

    # Evaluation
    python neural_model/evaluate.py \
        --checkpoint result/neural_model/best_model.pt
"""

from .model import (
    ADAPTClassifier,
    build_model,
    AMSoftmaxLoss,
    MultiPrototypeHead,
    FiLMLayer,
)

from .dataset import (
    DrivingExplanationDataset,
    create_dataloaders,
    load_features_from_cache,
)

__all__ = [
    'ADAPTClassifier',
    'build_model',
    'AMSoftmaxLoss',
    'MultiPrototypeHead',
    'FiLMLayer',
    'DrivingExplanationDataset',
    'create_dataloaders',
    'load_features_from_cache',
]
