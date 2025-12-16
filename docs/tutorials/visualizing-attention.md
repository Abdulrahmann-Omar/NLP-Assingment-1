# Visualizing Attention

Learn to create and interpret attention heatmaps.

## Quick Visualization

```python
from seq2seq_attention_comparison import evaluate_sentence, plot_attention

# Translate and get attention
result, sentence, attention = evaluate_sentence(
    "I love programming",
    encoder, decoder,
    inp_lang, targ_lang,
    max_length_inp, max_length_targ
)

# Plot attention heatmap
plot_attention(
    attention[:len(result.split())+1, :len(sentence.split())],
    sentence,
    result,
    save_path='my_attention.png'
)
```

## Understanding the Heatmap

```
         Source (English)
         I   love   learning
       ┌───┬──────┬──────────┐
    eu │ █ │  ░   │     ░    │
       ├───┼──────┼──────────┤
   amo │ ░ │  █   │     ░    │  ← High attention on "love"
       ├───┼──────┼──────────┤
aprender│ ░ │  ░   │     █    │
       └───┴──────┴──────────┘
              Target (Portuguese)
```

- **Dark cells** = High attention (model focuses here)
- **Light cells** = Low attention
- **Diagonal pattern** = Word-by-word alignment
- **Spread pattern** = Context-dependent translation

## Comparing Both Models

```python
from seq2seq_attention_comparison import compare_attention_weights

result_b, result_l = compare_attention_weights(
    "how are you?",
    encoder_bahdanau, decoder_bahdanau,
    encoder_luong, decoder_luong,
    inp_lang, targ_lang,
    max_length_inp, max_length_targ,
    save_path='comparison.png'
)
```

## Custom Styling

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.matshow(attention, cmap='viridis')  # Try: 'Blues', 'Reds', 'magma'
plt.colorbar(im)
plt.xlabel('Source Tokens', fontsize=14)
plt.ylabel('Target Tokens', fontsize=14)
plt.savefig('styled_attention.png', dpi=300)
```

## Batch Visualization

```python
test_sentences = [
    "I am a student.",
    "She likes coffee.",
    "We are learning NLP."
]

for i, sent in enumerate(test_sentences):
    result, preprocessed, attention = evaluate_sentence(sent, ...)
    plot_attention(attention, preprocessed, result, 
                   save_path=f'attention_{i}.png')
```

---

[← Training Custom Data](training-custom-data.md) | [Next: Extending Models →](extending-models.md)
