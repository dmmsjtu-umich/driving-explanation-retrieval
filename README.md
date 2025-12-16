# Driving Explanation Retrieval on BDD-X

CSE 595 Final Project: Vision-Language Transformer for 50-way cluster classification on BDD-X driving explanations.

## Task

Given a driving video and action description (e.g., "the car slows down"), classify into one of 50 semantic explanation clusters (e.g., "red traffic light", "pedestrian crossing").

## Results

| Model | Top-1 | Top-3 | Top-5 | Macro-F1 |
|-------|-------|-------|-------|----------|
| Text-only (BERT) | 21.5% | - | - | - |
| Video+Text Concat | 23.8% | - | - | - |
| **VLTransformer (Ours)** | **33.92%** | **57.08%** | **67.29%** | **24.79%** |

## Architecture

- **Video Encoder**: 3D ResNet-18 with 2×2 spatial grid tokens (8 tokens per clip)
- **Text Encoder**: Frozen BERT-base for action narration
- **Fusion**: 4-layer Transformer with multi-head self-attention
- **Regularization**: Video dropout (10%), modality dropout (10%), label smoothing (0.1), entropy regularization

## Project Structure

```
driving-explanation-retrieval/
├── neural_model/          # Core model code
│   ├── model.py          # VLTransformer architecture
│   ├── dataset.py        # Data loading from TSV
│   └── train.py          # Training script
├── data_analysis/         # Clustering pipeline
│   └── cluster_reasons.py # K-Means + S-BERT clustering
├── baselines/             # Baseline models
└── examples/              # Data loading examples
```

## Usage

```bash
# Train
python neural_model/train.py \
    --data_json data_analysis/clustering_results/reasons_with_clusters.json \
    --frame_tsv_root /path/to/frame_tsv \
    --epochs 30 --batch_size 8 --lr 1e-4

# Evaluate
python neural_model/train.py --eval_only --checkpoint checkpoints/best.pt
```

## Dataset

- **BDD-X**: 26,521 video clips with action-explanation pairs
- **Splits**: Train 21,143 / Val 2,519 / Test 2,859
- **Clustering**: 50 semantic clusters via K-Means on Sentence-BERT embeddings

## Citation

```bibtex
@inproceedings{kim2018bddx,
  title={Textual explanations for self-driving vehicles},
  author={Kim, Jinkyu and Rohrbach, Anna and Darrell, Trevor and Canny, John and Rohrbach, Marcus},
  booktitle={ECCV},
  year={2018}
}
```

## License

MIT
