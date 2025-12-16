# 🧠 Seq2Seq Attention Mechanisms: Bahdanau vs Luong

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**A comprehensive comparison of Bahdanau (Additive) and Luong (Multiplicative) attention mechanisms in Neural Machine Translation**

[Overview](#overview) •
[Features](#-features) •
[Quick Start](#-quick-start) •
[Results](#-results) •
[Architecture](#-architecture)

</div>

---

## Overview

This project implements and compares two foundational attention mechanisms in sequence-to-sequence models for machine translation:

- **Bahdanau Attention** (Additive) - From ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473)
- **Luong Attention** (Multiplicative) - From ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025)

Both models are trained on an **English-Portuguese translation task** using identical hyperparameters, enabling a fair side-by-side comparison of their performance characteristics.

## ✨ Features

- 🔄 **Complete Seq2Seq Implementation** - Encoder-Decoder architecture with GRU cells
- 🎯 **Dual Attention Mechanisms** - Both Bahdanau and Luong attention fully implemented
- 📊 **Visual Comparisons** - Training curves and performance visualizations
- 📓 **Jupyter Notebook** - Interactive exploration with step-by-step execution
- 🐍 **Python Script** - Standalone executable for batch processing

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy matplotlib
```

### Run with Jupyter Notebook (Recommended)

```bash
jupyter notebook seq2seq_attention_comparison.ipynb
```

Then click **"Run All Cells"** from the Jupyter menu.

### Run as Python Script

```bash
python seq2seq_attention_comparison.py
```

**⏱️ Expected Runtime:** 30-60 minutes (depending on hardware)

## 📂 Project Structure

```
├── seq2seq_attention_comparison.ipynb   # 📓 Interactive Jupyter notebook
├── seq2seq_attention_comparison.py      # 🐍 Standalone Python script
├── NLP_10.ipynb                         # 📚 Additional NLP experiments
├── README.md                            # 📖 This file
└── README_COMPARISON.md                 # 📋 Detailed usage guide
```

## 🏗️ Architecture

### Bahdanau Attention (Additive)

```
score(sₜ, hᵢ) = vᵀ · tanh(W₁·sₜ + W₂·hᵢ)
```

- Uses a **feed-forward network** to compute alignment scores
- Concatenates encoder/decoder states before scoring
- More parameters, slightly higher computational cost

### Luong Attention (Multiplicative)

```
score(sₜ, hᵢ) = sₜᵀ · W · hᵢ
```

- Uses **dot product** with learned weight matrix
- Directly computes similarity between states
- Fewer parameters, faster computation

### Model Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Encoder   │────►│  Attention  │────►│   Decoder   │
│   (GRU)     │     │   Layer     │     │   (GRU)     │
└─────────────┘     └─────────────┘     └─────────────┘
      ▲                   │                    │
      │                   ▼                    ▼
  [Source]          [Context Vector]      [Target]
```

## ⚙️ Configuration

Customize training by modifying these parameters:

```python
BATCH_SIZE = 64        # Reduce if running out of memory
EMBEDDING_DIM = 256    # Word embedding dimensions
UNITS = 512            # Hidden layer size
NUM_EXAMPLES = 30000   # Dataset size
EPOCHS = 10            # Training iterations
```

## 📈 Results

After training, you'll receive:

| Metric | Bahdanau | Luong |
|--------|----------|-------|
| Final Loss | ~X.XX | ~X.XX |
| Training Time | Slower | Faster |
| Parameters | More | Fewer |

### Generated Outputs

- `training_comparison.png` - Training loss curves visualization

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Out of Memory** | Reduce `BATCH_SIZE` to 32 or `UNITS` to 256 |
| **Training Too Slow** | Reduce `EPOCHS` to 5 or `NUM_EXAMPLES` to 15000 |
| **No GPU Warning** | Normal - will use CPU (just slower) |

## 📚 Key Learnings

By running this comparison, you'll understand:

1. ✅ How different attention mechanisms perform on the same task
2. ✅ Training dynamics and convergence behavior
3. ✅ Computational trade-offs (speed vs. accuracy)
4. ✅ When to choose one attention type over another

## 📄 References

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong, M. T., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation*

## 👤 Author

**Zewail City of Science and Technology**  
NLP Course - 4th Year Assignment

---

<div align="center">

Made with ❤️ for NLP

</div>
