# ScratchFormer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-orange)](https://pytorch.org/)

A from-scratch implementation of the Transformer architecture from "Attention is All You Need" for English-to-German machine translation. Developed as a student project, this lightweight, educational model achieves ~10 BLEU on WMT subsets. It relies solely on PyTorch core principles, with no external ML libraries. Note: The model performs reasonably on short sentences but struggles with longer phrases, where translations can lose coherence and become hard to follow.

## 🚀 Features
- **Full Seq2Seq Pipeline**: Encoder-decoder with multi-head attention, positional encoding, and beam search generation.
- **Custom BPE Tokenizer**: Trained directly on WMT data for efficient subword handling.
- **Efficient Training**: Noam learning rate schedule, mixed precision (FP16), and gradient clipping for stability.
- **Eval Ready**: Built-in BLEU and perplexity metrics, plus CLI for inference.


## 📊 Results
| Dataset              | Model Size | Epochs | BLEU (Test) | PPL (Test) |
|----------------------|------------|--------|-------------|------------|
| WMT en-de (1.5M subset) | ~30M params | 48     | 10.0       | ~13.0     |

*Trained on RTX 3090. Full logs available in releases*

## 🛠️ Installation
1. **Clone the Repo**:  
   ```
   git clone https://github.com/jsonmen/scratchformer.git
   cd scratchformer
   ```

2. **Set Up Environment**:  
   ```
   uv sync  # Installs dependencies from pyproject.toml/uv.lock
   ```

3. **Prepare Data**:  
   ```
   mkdir data
   uv run src/dataset.py  # Downloads and processes WMT en-de subsets
   ```

4. **Prepare Model Weights**:  
   ```
   mkdir models
   cd models
   git clone https://huggingface.co/jsonmen/scratchformer  # Or download manually
   cd ..
   ```

5. **Train Tokenizer** (if needed):  
   ```
   uv run src/tokenizer.py
   ```

## 🎯 Usage
### Training
Run the full training pipeline with default hyperparameters:  
```
uv run src/train.py
```
- **Customization**: Tweak settings in `src/config.py` (e.g., epochs, batch size, learning rate scale).
- **Monitoring**: Launch TensorBoard for real-time logs:  
  ```
  tensorboard --logdir=runs
  ```
- **Checkpoints**: Automatically saved to `checkpoints/` every N steps and for best BLEU.

### Inference
Test translations via the CLI:  
```bash
uv run src/inference.py -i "Hello World!"
```
- **Output Example**:  
  ```bash
  Input: Hello World!
  Translation: Hallo Welt!
  ```
- **Options**: `--temperature 0.9 --top_k 70` for more diverse outputs.

## 📚 Code Structure (main files)
- `src/model.py`: Core Transformer components (attention, layers, generation).
- `src/train.py`: Training loop with scheduling and evaluation.
- `src/tokenizer.py`: BPE training and encoding/decoding utilities.
- `src/inference.py`: CLI for single-sentence translations.
- `src/dataset.py`: WMT data loading and preprocessing.
- `src/config.py`: Hyperparameters and constants.


## 📄 License
MIT License—feel free to use, modify, and share.

## Acknowledgments
Inspired by the original "Attention is All You Need" paper.

*(P.S. In few days I create file with code explanation)*