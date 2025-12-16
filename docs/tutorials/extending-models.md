# Extending the Models

Add new attention mechanisms or modify the architecture.

## Adding a New Attention Type

### Step 1: Create the Attention Class

```python
class ScaledDotAttention(tf.keras.layers.Layer):
    """
    Scaled Dot-Product Attention (Transformer-style)
    score = (Q · K^T) / √d_k
    """
    def __init__(self, units, name="scaled_dot_attention"):
        super().__init__(name=name)
        self.units = units
        self.scale = tf.math.sqrt(tf.cast(units, tf.float32))
    
    def call(self, query, values):
        # query: (batch, hidden_dim)
        # values: (batch, seq_len, hidden_dim)
        
        query_expanded = tf.expand_dims(query, 1)
        
        # Scaled dot product
        score = tf.matmul(query_expanded, values, transpose_b=True)
        score = score / self.scale
        score = tf.squeeze(score, 1)
        score = tf.expand_dims(score, -1)
        
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context, attention_weights
```

### Step 2: Use It

```python
scaled_attention = ScaledDotAttention(UNITS)
decoder_scaled = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, 
                         BATCH_SIZE, scaled_attention)
```

## Adding Multi-Head Attention

```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, units, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.units = units
        assert units % num_heads == 0
        
        self.depth = units // num_heads
        self.wq = tf.keras.layers.Dense(units)
        self.wk = tf.keras.layers.Dense(units)
        self.wv = tf.keras.layers.Dense(units)
        self.dense = tf.keras.layers.Dense(units)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, values):
        batch_size = tf.shape(query)[0]
        
        q = self.split_heads(self.wq(tf.expand_dims(query, 1)), batch_size)
        k = self.split_heads(self.wk(values), batch_size)
        v = self.split_heads(self.wv(values), batch_size)
        
        # Scaled dot-product attention per head
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat = tf.reshape(output, (batch_size, -1, self.units))
        
        return self.dense(tf.squeeze(concat, 1)), attention_weights
```

## Architecture Modifications

### Use LSTM Instead of GRU

```python
# In Encoder.__init__
self.lstm = tf.keras.layers.LSTM(
    enc_units,
    return_sequences=True,
    return_state=True
)

# In Encoder.call
output, hidden_state, cell_state = self.lstm(x, initial_state=[hidden, cell])
return output, (hidden_state, cell_state)
```

### Bidirectional Encoder

```python
self.gru = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(enc_units // 2, return_sequences=True, return_state=True)
)
```

---

[← Visualizing Attention](visualizing-attention.md) | [Back to Index →](../index.md)
