# Evaluation Functions API

## evaluate_sentence

Translate a sentence and return attention weights.

```python
result, preprocessed, attention_plot = evaluate_sentence(
    sentence, encoder, decoder, inp_lang, targ_lang,
    max_length_inp, max_length_targ
)
```

| Parameter | Description |
|-----------|-------------|
| `sentence` | Raw input string (e.g., "How are you?") |
| `encoder` | Trained encoder model |
| `decoder` | Trained decoder model |
| `inp_lang` | Source tokenizer |
| `targ_lang` | Target tokenizer |
| `max_length_inp` | Max source length |
| `max_length_targ` | Max target length |

| Returns | Type | Description |
|---------|------|-------------|
| `result` | str | Translated text |
| `preprocessed` | str | Preprocessed input |
| `attention_plot` | ndarray | Attention weights `(tgt_len, src_len)` |

### Example

```python
translation, sent, attn = evaluate_sentence(
    "I love you",
    encoder_bahdanau, decoder_bahdanau,
    inp_lang, targ_lang,
    max_length_inp, max_length_targ
)
print(translation)  # "eu te amo"
```

---

## plot_attention

Visualize attention weights as heatmap.

```python
plot_attention(attention, sentence, predicted_sentence, save_path=None)
```

| Parameter | Description |
|-----------|-------------|
| `attention` | Attention matrix `(tgt_len, src_len)` |
| `sentence` | Source sentence (for x-axis labels) |
| `predicted_sentence` | Translation (for y-axis labels) |
| `save_path` | Optional path to save figure |

---

## plot_training_comparison

Compare training curves from two models.

```python
plot_training_comparison(history_bahdanau, history_luong, save_path=None)
```

Creates a 2-panel figure:
1. **Left**: Loss over epochs (line chart)
2. **Right**: Final loss comparison (bar chart)

---

## compare_attention_weights

Side-by-side attention visualization from both models.

```python
result_b, result_l = compare_attention_weights(
    sentence,
    encoder_bahdanau, decoder_bahdanau,
    encoder_luong, decoder_luong,
    inp_lang, targ_lang,
    max_length_inp, max_length_targ,
    save_path=None
)
```

Creates a 2-panel figure showing attention heatmaps from both models.

---

## generate_summary

Create a text summary of comparison results.

```python
report = generate_summary(history_bahdanau, history_luong, save_path=None)
```

| Returns | Description |
|---------|-------------|
| `report` | Formatted string with metrics |

Output includes:
- Number of epochs
- Initial/final loss for each model
- Loss reduction per model
- Winner declaration

---

[← Training Functions](training-functions.md) | [Back to Index →](../index.md)
