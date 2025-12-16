# Seq2Seq Architecture

The Sequence-to-Sequence (Seq2Seq) architecture is the foundation for neural machine translation.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Seq2Seq with Attention                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "How are you?"                                      │
│         ↓                                                   │
│  ┌──────────────────────────────────┐                       │
│  │           ENCODER                │                       │
│  │  [Embed] → [GRU] → [GRU] → [GRU] │                       │
│  │              ↓        ↓       ↓   │                       │
│  │             h₁       h₂      h₃   │ ← All hidden states  │
│  └──────────────────────────────────┘                       │
│                       ↓                                     │
│              [Attention Layer]                              │
│                       ↓                                     │
│  ┌──────────────────────────────────┐                       │
│  │           DECODER                │                       │
│  │  [Embed] → [GRU] → [Dense]       │                       │
│  │              ↓                   │                       │
│  │          "Como"                  │                       │
│  └──────────────────────────────────┘                       │
│                                                             │
│  Output: "Como você está?"                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Encoder

Processes the source sequence and produces hidden representations.

```python
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(units, return_sequences=True, return_state=True)
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state  # output = all h_i, state = final h
```

**Key Points:**
- `return_sequences=True`: Return all hidden states (for attention)
- `return_state=True`: Return final state (to initialize decoder)

### 2. Attention Layer

Computes context vector by weighting encoder states.

```python
# Give decoder access to all encoder states
context, weights = attention(decoder_hidden, encoder_outputs)
```

### 3. Decoder

Generates output sequence one token at a time.

```python
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, attention_layer):
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(units, return_state=True)
        self.fc = Dense(vocab_size)
        self.attention = attention_layer
    
    def call(self, x, hidden, encoder_output):
        # Get context from attention
        context, attn_weights = self.attention(hidden, encoder_output)
        
        # Embed input
        x = self.embedding(x)
        
        # Concatenate context with embedding
        x = concat([context, x], axis=-1)
        
        # Run through GRU
        output, state = self.gru(x)
        
        # Project to vocabulary
        output = self.fc(output)
        
        return output, state, attn_weights
```

## Training: Teacher Forcing

During training, we use the **ground truth** previous token as decoder input (not the predicted token).

```python
for t in range(1, target_length):
    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
    
    loss += compute_loss(target[:, t], predictions)
    
    # Teacher forcing: use actual target as next input
    dec_input = target[:, t]  # NOT argmax(predictions)
```

**Benefits:**
- Faster convergence
- More stable training

**Drawbacks:**
- Exposure bias (training/inference mismatch)

## GRU vs LSTM

This implementation uses **GRU** (Gated Recurrent Unit):

| Aspect | GRU | LSTM |
|--------|-----|------|
| Gates | 2 (reset, update) | 3 (forget, input, output) |
| Parameters | Fewer | More |
| Speed | Faster | Slower |
| Performance | Similar | Similar |

## Data Flow Summary

```
1. Input tokens → Embedding → Encoder GRU → All hidden states
2. Decoder state + Encoder states → Attention → Context vector
3. Context + Previous token embedding → Decoder GRU → Output probabilities
4. Repeat steps 2-3 for each output token
```

---

[← Luong Attention](luong-attention.md) | [Back to Index →](../index.md)
