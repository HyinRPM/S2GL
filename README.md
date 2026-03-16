# S2GL: Structure-Function Graph Learning for Brain Network Classification

A multimodal graph neural network framework for brain network classification, leveraging both structural and functional connectivity graphs.

## Overview

S2GL is a deep learning framework designed for brain network analysis and classification. It integrates structural connectivity (SC) and functional connectivity (FC) graphs through cross-attention mechanisms and multi-modal feature aggregation strategies.

## Key Features

- **Multimodal Learning**: Jointly processes structural and functional brain connectivity graphs
- **Cross-Attention Interaction (CAI)**: Enables information exchange between SC and FC modalities
- **Multi-modal Feature Aggregation (MFA)**: Fuses features from multiple modalities using mean, max, and sum pooling
- **Graph Autoencoder**: Uses GCN-based encoder with link prediction decoder for representation learning
- **Multi-level Pooling**: TopK pooling mechanism for selecting salient brain regions
- **Edge Masking**: Data augmentation through random edge masking

## Project Structure

```
S2GL/
├── run.py              # Main entry point with training loop and cross-validation
├── ca_mgae.py          # Core CA_MGAE2 model and FCNN classifier
├── mgae.py             # GCN encoder (GCN_mgaev3) and Link Prediction decoder (LPDecoder)
├── cai.py              # Cross-Attention Interaction module
├── mfa.py              # Multi-modal Feature Aggregation modules
├── multi_pooling.py    # Multi-graph pooling with PoolScore
├── train_test.py       # Training and testing functions
├── multi_dataload.py   # Multimodal dataset loader
└── utils.py            # Utility functions (edge masking, loss functions)
```

## Architecture

```
Input: Structural Graph (SC) + Functional Graph (FC)
                    │
                    ▼
        ┌───────────────────────┐
        │   GCN Encoder (MGAE)  │
        │   with CAI modules    │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Multi-level Pooling  │
        │   (TopK Selection)    │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   MFA Aggregation     │
        │ (Mean/Max/Sum Pool)   │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   FCNN Classifier     │
        └───────────────────────┘
                    │
                    ▼
              Classification
```

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- scikit-learn
- NumPy
- Matplotlib

```bash
pip install torch torch_geometric scikit-learn numpy matplotlib
```

## Data Format

The model expects brain graph data in pickle format:

```python
# structure.pkl - List of structural connectivity graphs
# function.pkl - List of functional connectivity graphs
```

Each graph should be a PyTorch Geometric `Data` object with:
- `x`: Node features [num_nodes, num_features]
- `edge_index`: Edge indices [2, num_edges]

## Usage

### Basic Training

```bash
python run.py
```

### Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_nodes` | int | 90 | Number of brain regions (nodes) |
| `--num_features` | int | 90 | Dimension of node features |
| `--num_classes` | int | 2 | Number of classification classes |
| `--encoder_layers` | int | 2 | Number of GCN encoder layers |
| `--decoder_layers` | int | 2 | Number of decoder layers |
| `--encoder_dim` | int | 16 | Hidden dimension of encoder |
| `--decoder_dim` | int | 256 | Hidden dimension of decoder |
| `--num_fcnnlayers` | int | 4 | Number of FCNN classifier layers |
| `--fcnn_dim` | int | 256 | Hidden dimension of FCNN |
| `--dropout` | float | 0.5 | Dropout rate |
| `--batch_size` | int | 16 | Training batch size |
| `--lr` | float | 0.005 | Learning rate |
| `--epoches` | int | 500 | Number of training epochs |
| `--mask_ratio` | float | 0.7 | Edge masking ratio |
| `--pool_ratio` | float | 0.25 | Pooling ratio for TopK |
| `--mask_type` | str | 'um' | Mask type: 'um' (undirected) or 'dm' (directed) |

## Model Components

### 1. GCN Encoder (mgae.py)
Multi-layer GCN encoder with cross-attention interaction between SC and FC features at each layer.

### 2. Cross-Attention Interaction (cai.py)
Implements co-attention mechanism to compute attention weights between structural and functional features.

### 3. Multi-modal Feature Aggregation (mfa.py)
Aggregates features using multiple strategies:
- `MFA2`: Concatenates mean, max, and sum pooled features

### 4. Multi-Graph Pooling (multi_pooling.py)
Learns unified pooling scores across both modalities and selects top-k important nodes.

### 5. Link Prediction Decoder (mgae.py)
Cross-layer decoder for link prediction as an auxiliary task.

## Training Strategy

- **10-fold Cross-Validation**: Repeated 3 times for robust evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Loss**: Cross-entropy + Link prediction loss + L2 regularization

## Output

The model outputs:
- Per-fold accuracy, precision, recall, F1 scores
- Mean metrics across all folds and runs

## Citation

If you use this code, please cite the related paper.

## License

This project is for research purposes.

## Contact

For questions or issues, please open an issue in the repository.
