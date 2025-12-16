# Luong Attention (Multiplicative Attention)

> **Paper**: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) (Luong et al., 2015)

## Key Innovation

Luong proposed simpler scoring functions and introduced the distinction between **global** and **local** attention. The multiplicative approach is computationally more efficient.

## Scoring Variants

### 1. Dot Product (`dot`)
```
score(sₜ, hᵢ) = sₜᵀ · hᵢ
```
- **Fastest** computation
- Requires same dimensions for encoder/decoder
- No learnable parameters

### 2. General (`general`) - Default
```
score(sₜ, hᵢ) = sₜᵀ · W · hᵢ
```
- Learned weight matrix W
- Handles dimension mismatch
- Most commonly used

### 3. Concat (`concat`)
```
score(sₜ, hᵢ) = vᵀ · tanh(W · [sₜ; hᵢ])
```
- Similar to Bahdanau
- Concatenates states before projection

## Comparison Table

| Variant | Formula | Parameters | Speed |
|---------|---------|------------|-------|
| Dot | `sᵀh` | 0 | ⚡ Fastest |
| General | `sᵀWh` | d × d | 🔹 Medium |
| Concat | `vᵀtanh(W[s;h])` | 2d + 1 | 🔸 Slowest |

## Implementation

```python
class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units, score_type='general'):
        super().__init__()
        self.score_type = score_type
        
        if score_type == 'general':
            self.W = tf.keras.layers.Dense(units, use_bias=False)
        elif score_type == 'concat':
            self.W = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        query_expanded = tf.expand_dims(query, 1)
        
        if self.score_type == 'dot':
            score = tf.matmul(query_expanded, values, transpose_b=True)
            score = tf.squeeze(score, 1)
            score = tf.expand_dims(score, -1)
            
        elif self.score_type == 'general':
            score = tf.matmul(query_expanded, self.W(values), transpose_b=True)
            score = tf.squeeze(score, 1)
            score = tf.expand_dims(score, -1)
            
        elif self.score_type == 'concat':
            query_tiled = tf.tile(query_expanded, [1, tf.shape(values)[1], 1])
            concat = tf.concat([query_tiled, values], axis=-1)
            score = self.V(tf.nn.tanh(self.W(concat)))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context, attention_weights
```

## Global vs Local Attention

### Global Attention (Implemented)
- Attends to **all** source positions
- More flexible but slower for long sequences

### Local Attention (Not Implemented)
- Attends to **window** around predicted position
- Faster but requires position prediction

```
Global: [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]  ← All states
Local:       [h₃, h₄, h₅]                  ← Window only
```

## When to Use Each Variant

| Use Case | Recommended |
|----------|-------------|
| Same encoder/decoder dimensions | `dot` |
| Different dimensions | `general` |
| Maximum expressiveness | `concat` |
| Speed-critical applications | `dot` |

---

[← Bahdanau Attention](bahdanau-attention.md) | [Next: Seq2Seq Architecture →](seq2seq-architecture.md)
