"""
Seq2Seq Model Comparison: Bahdanau vs Luong Attention

This script implements and compares two attention mechanisms:
1. Bahdanau Attention (Additive Attention)
2. Luong Attention (Multiplicative Attention)

For English-Portuguese machine translation task.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import re
import os
import io
import time
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def unicode_to_ascii(s):
    """Convert unicode to ASCII"""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    """Preprocess a sentence for tokenization"""
    w = unicode_to_ascii(w.lower().strip())
    
    # Add space between word and punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    
    # Add start and end tokens
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples=None):
    """Load and preprocess the dataset"""
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] 
                  for l in lines[:num_examples]]
    
    return zip(*word_pairs)

def tokenize(lang):
    """Tokenize a language corpus"""
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    """Load the dataset and create tokenizers"""
    targ_lang, inp_lang = create_dataset(path, num_examples)
    
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

# =============================================================================
# Attention Mechanisms
# =============================================================================

class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau Attention (Additive Attention)
    
    score(s_t, h_i) = v^T * tanh(W_s * s_t + W_h * h_i)
    """
    def __init__(self, units, name="bahdanau_attention"):
        super(BahdanauAttention, self).__init__(name=name)
        self.W1 = tf.keras.layers.Dense(units, name="W1")
        self.W2 = tf.keras.layers.Dense(units, name="W2")
        self.V = tf.keras.layers.Dense(1, name="V")
    
    def call(self, query, values):
        """
        Args:
            query: Decoder hidden state (batch, hidden_dim)
            values: Encoder outputs (batch, seq_len, hidden_dim)
        Returns:
            context_vector: (batch, hidden_dim)
            attention_weights: (batch, seq_len, 1)
        """
        # Expand query dimensions for broadcasting
        query_with_time_axis = tf.expand_dims(query, 1)  # (batch, 1, hidden_dim)
        
        # Score calculation
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))  # (batch, seq_len, 1)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)
        
        # Context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_dim)
        
        return context_vector, attention_weights


class LuongAttention(tf.keras.layers.Layer):
    """
    Luong Attention (Multiplicative Attention)
    
    Three scoring functions:
    - 'dot': score(s_t, h_i) = s_t^T * h_i
    - 'general': score(s_t, h_i) = s_t^T * W_a * h_i
    - 'concat': score(s_t, h_i) = v^T * tanh(W_a * [s_t; h_i])
    """
    def __init__(self, units, score_type='general', name="luong_attention"):
        super(LuongAttention, self).__init__(name=name)
        self.score_type = score_type
        self.units = units
        
        if score_type == 'general':
            self.W = tf.keras.layers.Dense(units, use_bias=False, name="W_general")
        elif score_type == 'concat':
            self.W = tf.keras.layers.Dense(units, name="W_concat")
            self.V = tf.keras.layers.Dense(1, name="V_concat")
    
    def call(self, query, values):
        """
        Args:
            query: Decoder hidden state (batch, hidden_dim)
            values: Encoder outputs (batch, seq_len, hidden_dim)
        Returns:
            context_vector: (batch, hidden_dim)
            attention_weights: (batch, seq_len, 1)
        """
        # Expand query dimensions
        query_with_time_axis = tf.expand_dims(query, 1)  # (batch, 1, hidden_dim)
        
        if self.score_type == 'dot':
            # Dot product attention
            score = tf.matmul(query_with_time_axis, values, transpose_b=True)
            score = tf.squeeze(score, axis=1)  # (batch, seq_len)
            score = tf.expand_dims(score, axis=-1)  # (batch, seq_len, 1)
            
        elif self.score_type == 'general':
            # General (multiplicative) attention
            score = tf.matmul(query_with_time_axis, self.W(values), transpose_b=True)
            score = tf.squeeze(score, axis=1)  # (batch, seq_len)
            score = tf.expand_dims(score, axis=-1)  # (batch, seq_len, 1)
            
        elif self.score_type == 'concat':
            # Concat attention (similar to Bahdanau but with concat)
            query_tiled = tf.tile(query_with_time_axis, [1, tf.shape(values)[1], 1])
            concat = tf.concat([query_tiled, values], axis=-1)
            score = self.V(tf.nn.tanh(self.W(concat)))  # (batch, seq_len, 1)
        
        # Attention weights via softmax
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)
        
        # Context vector as weighted sum
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_dim)
        
        return context_vector, attention_weights


# =============================================================================
# Encoder
# =============================================================================

class Encoder(tf.keras.Model):
    """Encoder with GRU/LSTM"""
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_sz
        return tf.zeros((batch_size, self.enc_units))


# =============================================================================
# Decoder
# =============================================================================

class Decoder(tf.keras.Model):
    """Decoder with configurable attention mechanism"""
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_layer):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = attention_layer
    
    def call(self, x, hidden, enc_output):
        # Get context vector from attention
        context_vector, attention_weights = self.attention(hidden, enc_output)
        
        # Embed the input
        x = self.embedding(x)
        
        # Concatenate context vector and embedding
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # Pass through GRU
        output, state = self.gru(x)
        
        # Reshape output
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # Dense layer for vocabulary projection
        x = self.fc(output)
        
        return x, state, attention_weights


# =============================================================================
# Training Functions
# =============================================================================

def loss_function(real, pred, loss_object):
    """Compute masked loss"""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, 
               optimizer, loss_object):
    """Single training step"""
    loss = 0
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * inp.shape[0], 1)
        
        # Teacher forcing
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions, loss_object)
            dec_input = tf.expand_dims(targ[:, t], 1)
    
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss


def train_model(encoder, decoder, dataset, targ_lang, epochs, batch_size, 
                steps_per_epoch, model_name="model"):
    """Train a model and return training history"""
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    
    # Checkpoint setup
    checkpoint_dir = f'./training_checkpoints_{model_name}'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                      encoder=encoder,
                                      decoder=decoder)
    
    history = {'loss': []}
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} Model")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        start = time.time()
        
        enc_hidden = encoder.initialize_hidden_state(batch_size)
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder,
                                   targ_lang, optimizer, loss_object)
            total_loss += batch_loss
            
            if batch % 100 == 0:
                print(f'  Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
        epoch_loss = total_loss / steps_per_epoch
        history['loss'].append(epoch_loss.numpy())
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print(f'Epoch {epoch + 1} Loss {epoch_loss:.4f} Time: {time.time() - start:.2f}s')
    
    return history


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_sentence(sentence, encoder, decoder, inp_lang, targ_lang, 
                      max_length_inp, max_length_targ):
    """Evaluate a single sentence"""
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index.get(i, 0) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post'
    )
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    hidden = encoder.initialize_hidden_state(1)
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )
        
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()[:max_length_inp]
        
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        if targ_lang.index_word.get(predicted_id, '') == '<end>':
            return result, sentence, attention_plot
        
        result += targ_lang.index_word.get(predicted_id, '') + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence, save_path=None):
    """Plot attention weights"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence.split(' '), fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence.split(' '), fontdict=fontdict)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# =============================================================================
# Comparison and Visualization
# =============================================================================

def plot_training_comparison(history_bahdanau, history_luong, save_path=None):
    """Plot training curves comparison"""
    plt.figure(figsize=(12, 5))
    
    # Loss comparison
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history_bahdanau['loss']) + 1)
    plt.plot(epochs, history_bahdanau['loss'], 'b-o', label='Bahdanau Attention', linewidth=2)
    plt.plot(epochs, history_luong['loss'], 'r-s', label='Luong Attention', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Final loss bar chart
    plt.subplot(1, 2, 2)
    final_losses = [history_bahdanau['loss'][-1], history_luong['loss'][-1]]
    colors = ['steelblue', 'coral']
    bars = plt.bar(['Bahdanau', 'Luong'], final_losses, color=colors, edgecolor='black')
    plt.ylabel('Final Loss', fontsize=12)
    plt.title('Final Training Loss', fontsize=14)
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    plt.show()


def compare_attention_weights(sentence, encoder_bahdanau, decoder_bahdanau,
                              encoder_luong, decoder_luong,
                              inp_lang, targ_lang, max_length_inp, max_length_targ,
                              save_path=None):
    """Compare attention weights from both models"""
    # Get attention from Bahdanau
    result_b, sent_b, attention_b = evaluate_sentence(
        sentence, encoder_bahdanau, decoder_bahdanau,
        inp_lang, targ_lang, max_length_inp, max_length_targ
    )
    
    # Get attention from Luong
    result_l, sent_l, attention_l = evaluate_sentence(
        sentence, encoder_luong, decoder_luong,
        inp_lang, targ_lang, max_length_inp, max_length_targ
    )
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bahdanau attention
    im1 = axes[0].matshow(attention_b[:len(result_b.split())+1, :len(sent_b.split())], 
                          cmap='viridis')
    axes[0].set_title('Bahdanau Attention', fontsize=14)
    axes[0].set_xlabel('Input (English)', fontsize=12)
    axes[0].set_ylabel('Output (Portuguese)', fontsize=12)
    
    # Luong attention
    im2 = axes[1].matshow(attention_l[:len(result_l.split())+1, :len(sent_l.split())], 
                          cmap='viridis')
    axes[1].set_title('Luong Attention', fontsize=14)
    axes[1].set_xlabel('Input (English)', fontsize=12)
    axes[1].set_ylabel('Output (Portuguese)', fontsize=12)
    
    plt.suptitle(f'Input: "{sentence}"', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention comparison to {save_path}")
    plt.show()
    
    return result_b, result_l


def generate_summary(history_bahdanau, history_luong, save_path=None):
    """Generate a summary report"""
    summary = []
    summary.append("=" * 60)
    summary.append("ATTENTION MECHANISM COMPARISON RESULTS")
    summary.append("=" * 60)
    summary.append("")
    summary.append("Training Summary:")
    summary.append("-" * 40)
    summary.append(f"Number of Epochs: {len(history_bahdanau['loss'])}")
    summary.append("")
    summary.append("Bahdanau Attention (Additive):")
    summary.append(f"  - Initial Loss: {history_bahdanau['loss'][0]:.4f}")
    summary.append(f"  - Final Loss:   {history_bahdanau['loss'][-1]:.4f}")
    summary.append(f"  - Loss Reduction: {history_bahdanau['loss'][0] - history_bahdanau['loss'][-1]:.4f}")
    summary.append("")
    summary.append("Luong Attention (Multiplicative):")
    summary.append(f"  - Initial Loss: {history_luong['loss'][0]:.4f}")
    summary.append(f"  - Final Loss:   {history_luong['loss'][-1]:.4f}")
    summary.append(f"  - Loss Reduction: {history_luong['loss'][0] - history_luong['loss'][-1]:.4f}")
    summary.append("")
    summary.append("-" * 40)
    winner = "Bahdanau" if history_bahdanau['loss'][-1] < history_luong['loss'][-1] else "Luong"
    diff = abs(history_bahdanau['loss'][-1] - history_luong['loss'][-1])
    summary.append(f"Better Performance: {winner} Attention (by {diff:.4f} loss)")
    summary.append("=" * 60)
    
    report = "\n".join(summary)
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"\nSaved summary to {save_path}")
    
    return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run the comparison"""
    print("Starting Seq2Seq Attention Mechanism Comparison")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    UNITS = 512
    NUM_EXAMPLES = 30000  # Use subset for faster training
    EPOCHS = 10
    
    # Download dataset if needed
    path_to_zip = tf.keras.utils.get_file(
        'por-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/por-eng.zip',
        extract=True
    )
    path_to_file = os.path.dirname(path_to_zip) + "/por-eng/por.txt"
    
    print(f"\nLoading dataset from: {path_to_file}")
    print(f"Number of examples: {NUM_EXAMPLES}")
    
    # Load and preprocess data
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
        path_to_file, NUM_EXAMPLES
    )
    
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
    print(f"Max input length: {max_length_inp}")
    print(f"Max target length: {max_length_targ}")
    
    # Train/test split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
        train_test_split(input_tensor, target_tensor, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(input_tensor_train)}")
    print(f"Validation samples: {len(input_tensor_val)}")
    
    # Vocabulary sizes
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    print(f"Input vocabulary size: {vocab_inp_size}")
    print(f"Target vocabulary size: {vocab_tar_size}")
    
    # Create dataset
    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    # =================================================================
    # Model 1: Bahdanau Attention
    # =================================================================
    print("\n" + "=" * 60)
    print("Creating Bahdanau Attention Model")
    print("=" * 60)
    
    encoder_bahdanau = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    bahdanau_attention = BahdanauAttention(UNITS)
    decoder_bahdanau = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE, 
                               bahdanau_attention)
    
    history_bahdanau = train_model(
        encoder_bahdanau, decoder_bahdanau, dataset, targ_lang,
        EPOCHS, BATCH_SIZE, steps_per_epoch, model_name="bahdanau"
    )
    
    # =================================================================
    # Model 2: Luong Attention (General)
    # =================================================================
    print("\n" + "=" * 60)
    print("Creating Luong Attention Model")
    print("=" * 60)
    
    encoder_luong = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    luong_attention = LuongAttention(UNITS, score_type='general')
    decoder_luong = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE, 
                            luong_attention)
    
    history_luong = train_model(
        encoder_luong, decoder_luong, dataset, targ_lang,
        EPOCHS, BATCH_SIZE, steps_per_epoch, model_name="luong"
    )
    
    # =================================================================
    # Comparison and Visualization
    # =================================================================
    print("\n" + "=" * 60)
    print("Generating Comparison Results")
    print("=" * 60)
    
    # Plot training comparison
    plot_training_comparison(
        history_bahdanau, history_luong,
        save_path='training_comparison.png'
    )
    
    # Compare attention on sample sentences
    test_sentences = [
        "how are you?",
        "this is my house.",
        "i love learning languages."
    ]
    
    for i, sentence in enumerate(test_sentences):
        result_b, result_l = compare_attention_weights(
            sentence, encoder_bahdanau, decoder_bahdanau,
            encoder_luong, decoder_luong,
            inp_lang, targ_lang, max_length_inp, max_length_targ,
            save_path=f'attention_comparison_{i+1}.png'
        )
        print(f"\nInput: {sentence}")
        print(f"  Bahdanau translation: {result_b}")
        print(f"  Luong translation: {result_l}")
    
    # Generate summary report
    generate_summary(
        history_bahdanau, history_luong,
        save_path='results_summary.txt'
    )
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - training_comparison.png: Training loss curves")
    print("  - attention_comparison_*.png: Attention weight visualizations")
    print("  - results_summary.txt: Summary report")


if __name__ == "__main__":
    main()
