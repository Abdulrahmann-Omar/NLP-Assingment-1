# Attention Mechanisms in Neural Machine Translation

## The Problem: Information Bottleneck

In vanilla Seq2Seq models, the encoder compresses the entire source sentence into a **single fixed-size vector**. This creates an information bottleneck:

```
Source: "The quick brown fox jumps over the lazy dog"
        ↓ (Encoder)
   [Fixed-size vector] ← All information compressed here!
        ↓ (Decoder)
Output: "Le renard brun rapide..."
```

**Problems:**
- Long sentences lose information
- Early tokens "forgotten" by later encoding
- No way to focus on relevant source words

## The Solution: Attention

Attention allows the decoder to **look back** at all encoder hidden states and focus on the most relevant ones for each output token.

```
Instead of:  Encoder → [single vector] → Decoder

Attention:   Encoder → [all hidden states h₁, h₂, ..., hₙ]
                              ↓
                       Attention weights (which states matter?)
                              ↓
                       Context vector (weighted sum)
                              ↓
                           Decoder
```

## General Attention Formula

For each decoder step t:

1. **Score**: Compute alignment scores between decoder state `sₜ` and each encoder state `hᵢ`
   ```
   score(sₜ, hᵢ) = f(sₜ, hᵢ)  ← Different mechanisms use different f()
   ```

2. **Normalize**: Convert scores to probabilities
   ```
   αₜᵢ = softmax(score(sₜ, hᵢ))
   ```

3. **Weight**: Compute context as weighted sum
   ```
   cₜ = Σᵢ αₜᵢ · hᵢ
   ```

## Attention Variants

| Type | Authors | Scoring Function | Key Idea |
|------|---------|------------------|----------|
| Additive (Bahdanau) | Bahdanau et al., 2014 | `vᵀtanh(W₁s + W₂h)` | Feed-forward network |
| Multiplicative (Luong) | Luong et al., 2015 | `sᵀWh` | Bilinear product |
| Dot Product | Luong et al., 2015 | `sᵀh` | Simple dot product |
| Scaled Dot | Vaswani et al., 2017 | `sᵀh / √d` | Transformers |

## Why Attention Works

1. **Alignment Learning**: Model learns which source words map to which target words
2. **Gradient Flow**: Direct connections help gradients flow to early encoder states
3. **Interpretability**: Attention weights visualize model's "focus"

---

[← Back to Index](../index.md) | [Next: Bahdanau Attention →](bahdanau-attention.md)
