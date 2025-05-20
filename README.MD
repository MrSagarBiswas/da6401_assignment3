# Bangla Transliteration Seq2Seq Models

This repository implements two character‐level sequence‐to‐sequence models for transliterating Roman‐script inputs into Bangla (Devanagari) script:
1. **Vanilla_RNN** – a basic encoder–decoder model without attention  
2. **Attention_RNN** – an encoder–decoder model augmented with Bahdanau (additive) attention

Both variants are written in PyTorch. You will find code to train on a sampled Bangla transliteration dataset, evaluate on held‐out dev/test sets, perform beam‐search decoding, and visualize attention weights for random samples.



## Repository Structure

```

.
├── Dataset
│   ├── bn.translit.sampled.train.tsv
│   ├── bn.translit.sampled.dev.tsv
│   └── bn.translit.sampled.test.tsv
│
├── Vanilla_RNN
│   ├── best_model_vanilla.pt
│   ├── predictions_vanilla.tsv
│   ├── vanilla_rnn_on_test_data.ipynb
│   ├── vanilla_rnn_predictions.png
│   └── Vanilla_RNN.ipynb
│
├── Attention_RNN
│   ├── best_model_attn.pt
│   ├── predictions_attention.tsv
│   ├── attention_rnn.ipynb
│   ├── attention_rnn_on_test_data.ipynb
│   ├── attention_visualization.ipynb
│   ├── attention_heatmap.png
│   └── Attention_RNN.ipynb
│
└── README.md
```

- **Dataset:**  
  - `bn.translit.sampled.train.tsv` – Training split (each line: `<Devanagari>\t<Roman>`)  
  - `bn.translit.sampled.dev.tsv`  – Development (dev) split  
  - `bn.translit.sampled.test.tsv` – Held‐out test split  

- **Vanilla_RNN:**  
  - `vanilla_rnn.py` – Core model definition and training loop for the vanilla Seq2Seq RNN.  
  - `vanilla_rnn_on_test_data.py` – Script for loading a saved vanilla RNN model, running on dev/test sets, computing word/char accuracy, and writing out `predictions_vanilla.tsv`.  
  - `best_model_vanilla.pt` – Example of a pre‐trained model checkpoint (if you check in one).  
  - `predictions_vanilla.tsv` – Example predictions file produced by `vanilla_rnn_on_test_data.py`.  
  - `vanilla_rnn_predictions.png` – (Optional) visualization of sample outputs.  
  - `Vanilla_RNN.ipynb` – a Colab‐style Jupyter notebook showing a step‐by‐step run of training and evaluation for the vanilla RNN.

- **Attention_RNN:**  
  - `attention_rnn.py` – Core model definition (encoder, Bahdanau attention, decoder) and training loop for the attention‐based Seq2Seq.  
  - `attention_rnn_on_test_data.py` – Script for loading a saved attention‐based model, running on dev/test sets (supports batched greedy decode and beam search), logging metrics to Weights & Biases, and writing `predictions_attention.tsv`.  
  - `attention_visualization.py` – Utility for plotting attention heatmaps for randomly chosen dev/test examples, saving figures like `attention_heatmap.png`.  
  - `best_model_attn.pt` – Example of a pre‐trained attention model checkpoint (if checked in).  
  - `predictions_attention.tsv` – Example predictions from running `attention_rnn_on_test_data.py`.  
  - `attention_heatmap.png` – Sample heatmap showing alignments for one transliteration pair.  
  - `Attention_RNN.ipynb` – Colab‐style notebook walking through training, evaluation, and visualization with attention.



## 1. Requirements

The code was developed and tested with:

- **Python 3.7+**  
- **PyTorch 1.7+** (or any recent 1.x release)  
- **torchvision** (if you plan to visualize via notebooks, although not strictly required)  
- **pandas**, **matplotlib**, **tqdm**  
- **wandb** (optional: only if you want to log to Weights & Biases)  

To install the core dependencies, you can run:

```bash
pip install torch torchvision pandas matplotlib tqdm
````

If you intend to use Weights & Biases (wandb) for logging:

```bash
pip install wandb
```

---

## 2. Data Format & Preprocessing

Each TSV file in **Dataset/** must have lines of the form:

```
<Devanagari_word>   <Roman_transliteration>   [<optional_third_column_ignored>]
```

For example:

```
অবস্থান    obosthan
আমি       ami
বাংলা     bangla
```

* **Column 0**: Gold Devanagari
* **Column 1**: Source Roman
* Anything beyond the second column is ignored.

Before running any scripts, verify that `bn.translit.sampled.*.tsv` contain valid `<dev>\t<rom>` pairs. If you create your own data, ensure you preserve that exact format.

---

## 3. Vanilla RNN Model

### 3.1 Model Architecture (`vanilla_rnn.py`)

* **CharVocab** – Builds two vocabularies:

  * `rom_char2idx` / `rom_idx2char` for Roman‐characters
  * `dev_char2idx` / `dev_idx2char` for Devanagari characters
  * Adds special tokens `<pad>`, `<sos>`, `<eos>`.

* **TransliterationDataset** (PyTorch `Dataset`)

  * Reads a TSV file and outputs `(roman_indices, devanagari_indices)` pairs.
  * Implements a custom `collate_fn` to pad each batch to the same max length.

* **Encoder** – An RNN (can be vanilla RNN, GRU or LSTM) that takes embedded Roman‐character sequences and returns hidden states.

* **Decoder** – Another RNN that takes the target Devanagari sequence (with `<sos>` prepended) and hidden state from encoder; uses teacher forcing during training.

* **Seq2Seq** – Wrapper that runs the encoder, transforms encoder hidden state to decoder hidden state dimensions, then runs the decoder to produce output‐logits.

#### Example usage (train/evaluate inside `vanilla_rnn.py`)

Within `if __name__ == "__main__":` or similar, typical arguments might be:

```python
python vanilla_rnn.py \
  --train_path ../Dataset/bn.translit.sampled.train.tsv \
  --dev_path   ../Dataset/bn.translit.sampled.dev.tsv \
  --cell_type  "LSTM" \
  --embed_dim  256 \
  --hidden_dim 512 \
  --num_layers 2 \
  --dropout    0.3 \
  --batch_size 128 \
  --num_epochs 15 \
  --learning_rate 1e-3 \
  --model_save_path best_model_vanilla.pt
```

The script should:

1. Build `CharVocab` over train+dev files.
2. Create `DataLoader` objects for train / dev using `TransliterationDataset` + `collate_fn`.
3. Instantiate `Encoder`, `Decoder`, and wrap in `Seq2Seq`.
4. Train using cross‐entropy loss (masked at padding), optionally logging to `wandb`.
5. After each epoch, evaluate on dev set and save the best checkpoint to `best_model_vanilla.pt`.

> **NOTE**: Adjust argument names to match your actual `argparse` flags in `vanilla_rnn.py`.

### 3.2 Generating Predictions (`vanilla_rnn_on_test_data.py`)

Once you have trained `best_model_vanilla.pt`, use:

```bash
python vanilla_rnn_on_test_data.py \
  --model_path   Vanilla_RNN/best_model_vanilla.pt \
  --test_path    Dataset/bn.translit.sampled.test.tsv \
  --output_path  Vanilla_RNN/predictions_vanilla.tsv \
  --beam_size    5        # set to 1 for greedy decode, >1 for beam search
  --max_len      32       # maximum target length for decoding
  --cell_type    "LSTM"   # must match what was used at training time
  --embed_dim    256      
  --hidden_dim   512      
  --num_layers   2        
  --dropout      0.3
```

* **What it does**:

  1. Loads `CharVocab` from the combination of train/dev/test (to ensure consistent indexing).
  2. Loads the saved PyTorch model checkpoint (`best_model_vanilla.pt`).
  3. For each input Roman string in `bn.translit.sampled.test.tsv`, runs greedy or beam decode to produce a predicted Devanagari string.
  4. Writes results (one line per test input) into `predictions_vanilla.tsv` with columns:

     ```
     <Roman_input> <Predicted_Bengali_word> <Char_Accuracy> <Word_Correct>
     ```
  5. Prints overall test‐set word/character accuracy to stdout (and optionally logs to `wandb`).

* **Example output** (`predictions_vanilla.tsv`):

  ```
  আমি      ami     আমি      1.0    1.0
  kuwe     kuwe    কুওয়     0.0    0.75
  bangali  bangali বাংলা     1.0    1.0
  ...
  ```

* After running, you can plot or inspect `vanilla_rnn_predictions.png` (if provided) to see random sample‐by‐sample visual comparisons.

---

## 4. Attention RNN Model

### 4.1 Model Architecture (`attention_rnn.py`)

This file is almost identical in structure to `vanilla_rnn.py` but introduces:

* **BahdanauAttention** – A module that, given the decoder’s last hidden state and the full sequence of encoder outputs, computes attention weights over all source time steps, then returns a context vector (weighted sum of encoder outputs).

* **Decoder** – At each target timestep, the decoder:

  1. Embeds the previous token
  2. Calls `attention(hidden, encoder_outputs)` to get a context vector of dimension `(batch, hidden_size)`
  3. Concatenates `[embedding ∥ context]` and feeds into the RNN cell
  4. Projects the RNN’s output to vocabulary logits

* **Seq2Seq** – Annotates how to pass `encoder_outputs` into the decoder and handle hidden/cell bridging.

All other classes (`CharVocab`, `TransliterationDataset`, `Encoder`) remain functionally the same, except your decoder now expects one extra input (`encoder_outputs`) at each forward pass.

#### Example usage (train/evaluate inside `attention_rnn.py`)

```bash
python attention_rnn.py \
  --train_path ../Dataset/bn.translit.sampled.train.tsv \
  --dev_path   ../Dataset/bn.translit.sampled.dev.tsv \
  --cell_type  "LSTM" \
  --embed_dim  256 \
  --hidden_dim 512 \
  --num_layers 2 \
  --dropout    0.3 \
  --batch_size 128 \
  --num_epochs 15 \
  --learning_rate 1e-3 \
  --model_save_path best_model_attn.pt
```

During training, for each batch, the script should call:

```python
encoder_outputs, (h_n, c_n) = encoder(src_batch)
logits = decoder(
    tgt_batch,          # shifted gold Devanagari indices (with <sos>, <eos>)
    h_dec_initial,      # derived from h_n / c_n
    c_dec_initial,      # same for LSTM
    encoder_outputs,
    teacher_forcing_ratio=0.5,
)
loss = criterion(logits.transpose(1,2), tgt_batch)  # cross entropy over non‐padded tokens
```

### 4.2 Generating Predictions & Evaluation (`attention_rnn_on_test_data.py`)

Once you have trained `best_model_attn.pt`, execute:

```bash
python attention_rnn_on_test_data.py \
  --model_path    Attention_RNN/best_model_attn.pt \
  --test_path     Dataset/bn.translit.sampled.test.tsv \
  --output_path   Attention_RNN/predictions_attention.tsv \
  --beam_size     5         # or 1 for greedy
  --max_len       32        # max target length
  --cell_type     "LSTM"    # must match training
  --embed_dim     256
  --hidden_dim    512
  --num_layers    2
  --dropout       0.3
```

* **What it does**:

  1. Loads `CharVocab` built over train/dev/test.
  2. Restores the attention model from `best_model_attn.pt`.
  3. Runs **batched greedy decode** or beam‐search decode (depending on `--beam_size`) for each Roman input.
  4. Writes out `predictions_attention.tsv` with columns identical to the vanilla script:

     ```
     <Roman_input> <Predicted_Bengali_word> <Char_Accuracy> <Word_Correct>
     ```
  5. Prints overall test‐set metrics, logs to `wandb` (if configured).

### 4.3 Attention Visualization (`attention_visualization.py`)

To inspect *which source positions the decoder attends to at each target timstep*, run:

```bash
python attention_visualization.py \
  --model_path  Attention_RNN/best_model_attn.pt \
  --data_path   Dataset/bn.translit.sampled.dev.tsv \
  --out_dir     Attention_RNN/heatmaps \
  --num_samples 9 \
  --cell_type   "LSTM" \
  --embed_dim   256 \
  --hidden_dim  512 \
  --num_layers  2 \
  --dropout     0.3
```

* **What it does**:

  1. Loads the same `CharVocab` + trained model.
  2. Randomly selects `num_samples` pairs from the specified data file.
  3. For each selected pair:

     * Runs a **single‐instance beam (or greedy) decode** while collecting attention weights at each decoder step.
     * Uses `matplotlib` to plot a heatmap:

       * X‐axis = source Roman characters (`[<sos>, r1, r2, …, <eos>]`)
       * Y‐axis = predicted (generated) target characters (`[<sos>, d1, d2, …, <eos>]`)
       * Color intensity shows attention weight α<sub>t,s</sub> between decoder step *t* and encoder step *s*.
  4. Saves each heatmap as a PNG into `Attention_RNN/heatmaps/` (e.g., `heatmap_0.png`, …).
  5. Optionally aggregates them in a single figure `attention_heatmap.png`.

After running, you can open the saved PNG(s) to see how the model “aligns” each Roman letter to Bangla letters. In the repo’s root, an example heatmap is already checked in as `Attention_RNN/attention_heatmap.png`.

---

## 5. Quick Start / Example Workflow

Below is a **minimal set of commands** to reproduce:

1. **Clone the repository**

   ```bash
   git clone https://github.com/MrSagarBiswas/da6401_assignment3.git
   cd da6401_assignment3
   ```

2. **Install dependencies**

   ```bash
   pip install torch torchvision pandas matplotlib tqdm wandb
   ```

3. **Train the Vanilla RNN**

   ```bash
   cd ../Vanilla_RNN
   ```
   Then run the Vanilla_RNN.ipynb notebook

   * Checkpoint `best_model_vanilla.pt` will be saved in `Vanilla_RNN/`

4. **Evaluate Vanilla RNN on Test Set**

    ```bash
   cd ../Vanilla_RNN
   ```
   Then run the Vanilla_RNN_on_Test_Data.ipynb notebook

   * The script writes `Vanilla_RNN/predictions_vanilla.tsv` and prints overall word/char accuracy.

5. **Train the Attention RNN**

   ```bash
   cd ../Attention_RNN
   ```
   Then run the Attention_RNN.ipynb notebook

   * Checkpoint `best_model_attn.pt` will appear in `Attention_RNN/`

6. **Evaluate Attention RNN on Test Set**

   ```bash
   Run the Attention_RNN_on_Test_Data.ipynb notebook
   ```

   * The script writes `Attention_RNN/predictions_attention.tsv` and prints overall metrics.

7. **Visualize Attention Weights**

   ```bash
   Run the Attention_Visualization.ipynb notebook
   ```

   * Nine heatmaps will be created under `Attention_RNN/`
   * Inspect `Attention_RNN/heatmaps.png` to see alignment patterns.

---

## 6. Code Organization & Key Files

Below is a more detailed listing of each top‐level file and how they interact.

### 6.1 `Dataset:`

* **bn.translit.sampled.train.tsv** / **bn.translit.sampled.dev.tsv** / **bn.translit.sampled.test.tsv**

  * Each line:

    ```
    <Devanagari_word>   <Roman_word>   <--->
    ```
  * Used by both Vanilla\_RNN and Attention\_RNN for building vocabulary and training.

### 6.2 `Vanilla_RNN:`

* **Vanilla_RNN.py**

  * Imports: `torch`, `torch.nn`, `torch.utils.data.Dataset`, `DataLoader`.
  * Defines:

    * `CharVocab` – builds char‐to‐index maps.
    * `TransliterationDataset` – wraps TSV → (input, target) tensors.
    * `Encoder` – RNN/GRU/LSTM encoder class.
    * `Decoder` – RNN/GRU/LSTM decoder class (teacher forcing optional).
    * `Seq2Seq` – wrapper that ties encoder + decoder and provides a `.forward(src, tgt, tf_ratio)` method.
  * Contains a training loop under `if __name__ == "__main__":` that:

    1. Parses command‐line args (paths, hidden sizes, learning rate, etc.).
    2. Builds `CharVocab` with train + dev paths.
    3. Creates `DataLoader` for train/dev.
    4. Instantiates `Encoder`, `Decoder`, `Seq2Seq`.
    5. Loops over epochs, batches → loss → `.backward()` → optimizer step.
    6. Evaluates on dev after each epoch, tracks best dev accuracy, saves best checkpoint.

* **Vanilla_RNN_on_Test_Data.ipynb**

  * Loads best checkpoint from `vanilla_rnn.py`.
  * Builds the same `CharVocab` (train+dev+test).
  * Provides a function to decode each input sequence (greedy or beam).
  * Iterates over test set, decodes, computes character‐level & word‐level accuracy, writes to `predictions_vanilla.tsv`.
  * Optionally logs metrics to `wandb`.
