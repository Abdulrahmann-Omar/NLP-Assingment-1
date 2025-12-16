# Training on Custom Data

Use your own translation dataset with this implementation.

## Data Format

Your dataset must be a text file with:
- One sentence pair per line
- Source and target separated by TAB (`\t`)

```
Hello	Olá
How are you?	Como você está?
This is a book.	Este é um livro.
```

## Step 1: Prepare Your Data

```python
# Create a file in the expected format
with open('my_dataset.txt', 'w', encoding='utf-8') as f:
    f.write("Hello\tBonjour\n")
    f.write("Goodbye\tAu revoir\n")
    # ... more pairs
```

## Step 2: Modify Data Loading

In `seq2seq_attention_comparison.py`, replace the dataset URL:

```python
# Original (Portuguese-English)
path_to_zip = tf.keras.utils.get_file(
    'por-eng.zip',
    origin='http://storage.googleapis.com/download.tensorflow.org/data/por-eng.zip',
    extract=True
)
path_to_file = os.path.dirname(path_to_zip) + "/por-eng/por.txt"

# Custom (use your file directly)
path_to_file = "path/to/my_dataset.txt"
```

## Step 3: Adjust Parameters

```python
# Based on your data size and complexity
NUM_EXAMPLES = len(open(path_to_file).readlines())
EMBEDDING_DIM = 256  # Increase for larger vocab
UNITS = 512          # Increase for complex languages
```

## Available Datasets

| Dataset | Languages | Size | URL |
|---------|-----------|------|-----|
| ManyThings | Multiple pairs | 100K+ | [Tab-delimited files](https://www.manythings.org/anki/) |
| Europarl | EU languages | 2M+ | [Europarl Corpus](https://www.statmt.org/europarl/) |
| OPUS | 90+ languages | Varies | [OPUS Collection](https://opus.nlpl.eu/) |

---

[← Back to Index](../index.md) | [Next: Visualizing Attention →](visualizing-attention.md)
