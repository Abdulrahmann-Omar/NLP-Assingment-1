# Attention Visualizations

Gallery of attention weight visualizations from the comparison experiments.

## Interpreting Attention Heatmaps

```
X-axis → Source tokens (English)
Y-axis → Target tokens (Portuguese)
Color  → Attention weight (darker = higher)
```

## Expected Patterns

### Diagonal Alignment
```
Source:  The   cat   sits
Target:  O     gato  senta
         [■ ░ ░]
         [░ ■ ░]
         [░ ░ ■]
```
Strong diagonal = word-for-word correspondence

### Reordering
```
Source:  I    love   you
Target:  Eu   te     amo
         [■ ░ ░]
         [░ ░ ■]   ← "te" attends to "you"
         [░ ■ ░]   ← "amo" attends to "love"
```
Non-diagonal = word order differs between languages

### Multi-word Attention
```
Source:  kick   the   bucket   (idiom)
Target:  morrer
         [░ ░ ████]  ← Attends to whole phrase
```
Spread attention = idiomatic or multi-word expressions

## Bahdanau vs Luong Patterns

| Aspect | Bahdanau Typical | Luong Typical |
|--------|------------------|---------------|
| Sharpness | Softer peaks | Sharper peaks |
| Spread | More distributed | More focused |
| Multi-word | Smoother blending | Discrete selection |

## Sample Sentences to Try

```python
test_sentences = [
    # Simple (expect diagonal)
    "this is a book",
    
    # Word reordering
    "I love you",
    
    # Longer with context
    "the quick brown fox jumps over the lazy dog",
    
    # Question
    "how are you today?"
]
```

## Generating Your Own Visualizations

```python
from seq2seq_attention_comparison import (
    evaluate_sentence, 
    compare_attention_weights
)

# After training both models
compare_attention_weights(
    "your sentence here",
    encoder_bahdanau, decoder_bahdanau,
    encoder_luong, decoder_luong,
    inp_lang, targ_lang,
    max_length_inp, max_length_targ,
    save_path='my_comparison.png'
)
```

## Generated Files

After running the comparison:

| File | Content |
|------|---------|
| `training_comparison.png` | Loss curves side-by-side |
| `attention_comparison_1.png` | "how are you?" |
| `attention_comparison_2.png` | "this is my house" |
| `attention_comparison_3.png` | "i love learning languages" |

---

[← Performance Analysis](performance-analysis.md) | [Back to Index →](../index.md)
