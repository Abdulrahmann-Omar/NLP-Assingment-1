# Bahdanau Attention (Additive Attention)

> **Paper**: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014)

## Key Innovation

Bahdanau introduced the first attention mechanism for NMT, allowing the decoder to **jointly learn to align and translate**. The model learns which source words to focus on for each target word.

## Mathematical Formulation

### Score Function (Additive/Concat)

```
score(sₜ, hᵢ) = vᵀ · tanh(W₁ · sₜ + W₂ · hᵢ)
```

Where:
- `sₜ` = Decoder hidden state at time t (query)
- `hᵢ` = Encoder hidden state at position i (value)
- `W₁`, `W₂` = Learned weight matrices
- `v` = Learned weight vector

### Full Computation

```python
# 1. Project query and keys
query_proj = W1 @ s_t          # Project decoder state
key_proj = W2 @ h_i            # Project encoder states

# 2. Combine through tanh
combined = tanh(query_proj + key_proj)

# 3. Compute scalar score
score = v.T @ combined

# 4. Normalize with softmax
alpha = softmax(scores)

# 5. Compute context vector
context = sum(alpha_i * h_i)
```

## Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│                  Bahdanau Attention              │
├──────────────────────────────────────────────────┤
│                                                  │
│   sₜ (decoder)    h₁, h₂, ..., hₙ (encoder)     │
│        │                  │                      │
│        ▼                  ▼                      │
│      [W₁]              [W₂]                      │
│        │                  │                      │
│        └──────┬───────────┘                      │
│               ▼                                  │
│           [tanh]                                 │
│               │                                  │
│               ▼                                  │
│             [vᵀ]                                 │
│               │                                  │
│               ▼                                  │
│          [softmax] → α₁, α₂, ..., αₙ            │
│               │                                  │
│               ▼                                  │
│     context = Σ αᵢ · hᵢ                         │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Implementation

```python
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)  # Query projection
        self.W2 = tf.keras.layers.Dense(units)  # Key projection
        self.V = tf.keras.layers.Dense(1)       # Score vector
    
    def call(self, query, values):
        # query: (batch, hidden_dim)
        # values: (batch, seq_len, hidden_dim)
        
        query_expanded = tf.expand_dims(query, 1)  # (batch, 1, hidden)
        
        # Additive score
        score = self.V(tf.nn.tanh(
            self.W1(query_expanded) + self.W2(values)
        ))  # (batch, seq_len, 1)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context, attention_weights
```

## Characteristics

| Aspect | Details |
|--------|---------|
| **Parameters** | 3 weight matrices (W₁, W₂, v) |
| **Complexity** | O(n × d × k) where k = attention units |
| **Pros** | More expressive, handles dimension mismatch |
| **Cons** | Slower than dot product |

---

[← Attention Mechanisms](attention-mechanisms.md) | [Next: Luong Attention →](luong-attention.md)
