# IMDB Sentiment Classification Using LSTM

## Project Overview

This project implements an NLP-based deep learning system for classifying IMDB movie reviews into **Positive** and **Negative** sentiments.

The project follows the given requirements from **Part 1 to Part 10**. The notebook contains the implementation, experiments, graphs, predictions, evaluation, and error analysis.

The final model used for test evaluation is a **Regularized LSTM** with **Dropout** and **Early Stopping**.

---

## Dataset

The project uses the IMDB Movie Reviews dataset from TensorFlow/Keras.

- Training reviews: **25,000**
- Testing reviews: **25,000**
- Total reviews: **50,000**
- Labels:
  - `0` → Negative
  - `1` → Positive
- Vocabulary size used: **10,000**

---

# Part 1 — Understand the Dataset

### 1. How many training and testing samples are available?

There are **25,000 training reviews** and **25,000 testing reviews**, giving **50,000 reviews in total**.

### 2. What does each element of `X_train` represent?

Each element of `X_train` represents one movie review as a sequence of **integer token IDs**.

### 3. Why are the reviews represented as integers instead of raw text?

The neural network needs numerical input. Therefore, each word is mapped to an integer token ID before it is passed to the model.

### 4. What does the vocabulary size mean?

A vocabulary size of **10,000** was selected. The model works with the selected vocabulary, while words outside it are represented by the unknown token:

```text
<UNK> → ID 2
```

The token IDs are identifiers and their numeric magnitude does not represent word meaning.

### 5. What is the distribution of positive and negative reviews?

The training data is balanced:

```text
Positive: 12,500
Negative: 12,500
```

## Review Length Statistics

| Statistic | Result |
|---|---:|
| Minimum review length | **11 tokens** |
| Maximum review length | **2,494 tokens** |
| Average review length | **238.71 tokens** |
| Median review length | **178 tokens** |
| 90th percentile | **467 tokens** |
| 95th percentile | **610 tokens** |

### Observation

The review-length histogram is right-skewed. Most reviews have moderate lengths, while a smaller number are much longer.

---

# Part 2 — Text Preprocessing

### 1. What vocabulary size was used?

The vocabulary size was set to:

```text
10,000
```

### 2. How were unknown words handled?

Words outside the selected vocabulary were represented using:

```text
<UNK> → ID 2
```

This allows the model to process words it does not have a separate vocabulary entry for.

### 3. Why is padding required?

The reviews have different lengths, but neural networks process training data in batches with consistent dimensions.

Therefore, every review was converted to a fixed length of **500 tokens**.

Short reviews were padded with zeros and long reviews were truncated.

### 4. What padding and truncation strategy was used?

The project used:

```python
padding="post"
truncating="post"
```

This means:

- Padding is added at the end.
- Extra tokens are removed from the end.

### 5. Why was `maxlen = 500` selected?

The value was selected after analyzing the review-length distribution.

The 90th percentile was **467 tokens**, so 500 was used as a practical limit that keeps most reviews while controlling sequence length.

Approximately **91.57%** of training reviews were within 500 tokens.

### 6. How was the training and validation data divided?

The original 25,000 training reviews were split into:

```text
Training:   20,000
Validation: 5,000
```

The split used:

```python
test_size=0.20
random_state=42
stratify=y_train
```

Final data shapes:

```text
X_train_final → (20000, 500)
X_val         → (5000, 500)
X_test_padded → (25000, 500)
```

### Why is padding important for batches?

Without padding, reviews would have different sequence lengths and could not be combined into a single fixed-size batch. Padding makes the sequence dimensions consistent.

---

# Part 3 — Build an Embedding + LSTM Model

## Architecture

```text
Input
   ↓
Embedding
   ↓
LSTM
   ↓
Dense
   ↓
Sigmoid Output
```

## Embedding

### What is an embedding?

An embedding converts each integer token ID into a dense numerical vector.

The project used:

```text
Vocabulary size = 10,000
Embedding dimension = 128
```

The embedding matrix therefore contains:

```text
10,000 × 128 = 1,280,000
```

trainable parameters.

### Why not directly feed word IDs into the LSTM?

A token ID such as `89` is only an identifier. The difference between IDs 89 and 90 does not mean that the words are semantically similar.

The Embedding layer learns useful vector representations during training.

## Why are RNNs useful for sequential data?

Text is sequential because word order matters. RNNs process tokens one after another and maintain information from earlier parts of the sequence.

## Vanishing-Gradient Problem

In long sequences, gradients can become very small while being passed through many recurrent steps. This makes it difficult for a basic RNN to learn long-range dependencies.

## Why was LSTM introduced?

LSTM was introduced to better preserve useful information over longer sequences and reduce the effect of the vanishing-gradient problem.

## LSTM Gates

- **Forget gate:** controls which old information should be retained or discarded.
- **Input gate:** controls which new information should be stored.
- **Output gate:** controls which information is exposed as the current output.

### Hidden State

The hidden state represents the information currently being passed forward from the LSTM.

### Cell State

The cell state acts as longer-term internal memory.

## Basic LSTM Parameters

| Layer | Output Shape | Parameters |
|---|---|---:|
| Embedding | `(None, 500, 128)` | 1,280,000 |
| LSTM | `(None, 64)` | 49,408 |
| Dense | `(None, 32)` | 2,080 |
| Dense | `(None, 1)` | 33 |
| **Total** | | **1,331,521** |

---

# Part 4 — Train the Model

## Training Configuration

The Basic LSTM was trained with:

```text
Loss function: Binary Crossentropy
Optimizer: Adam
Batch size: 64
Epochs: 10
```

### Why Binary Crossentropy?

The task is binary classification, so binary crossentropy is appropriate for measuring the difference between the actual label and predicted probability.

### Why Adam?

Adam updates the model weights using gradient information and adapts the update size for different parameters.

### What is the batch size?

A batch size of 64 means the model processes 64 reviews before performing a weight update.

### What is an epoch?

One epoch is one complete pass through the training dataset.

## Training Results

The recorded baseline run produced approximately:

```text
Training Accuracy:   96.06%
Validation Accuracy: 84.64%
```

The training accuracy became much higher than the validation accuracy.

### Graph Analysis

The training curves showed that the model continued learning the training data while validation performance stopped improving at the same rate.

### Model Behavior

**Overfitting was observed.**

The model learned the training data very well but did not achieve the same performance on unseen validation data.

---

# Part 5 — Improve the Model

Two architectures were trained and compared:

## Model 1 — Basic LSTM

```text
Embedding
   ↓
LSTM
   ↓
Dense
   ↓
Output
```

## Model 2 — Bidirectional LSTM

```text
Embedding
   ↓
Bidirectional LSTM
   ↓
Dense
   ↓
Output
```

## Model Comparison Results

The final comparison was calculated using the saved Basic LSTM and BiLSTM models on the same recreated validation set.

| Model | Accuracy | Precision | Recall | F1 Score | Training Time | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Basic LSTM | **84.64%** | **82.63%** | **87.72%** | **85.10%** | **88.45 sec** | **1,331,521** |
| Bidirectional LSTM | **85.74%** | **84.19%** | **88.00%** | **86.06%** | **148.56 sec** | **1,382,977** |

### Comparison Observation

The BiLSTM performed slightly better than the Basic LSTM on the validation set.

- Accuracy improved from **84.64% to 85.74%**.
- Precision improved from **82.63% to 84.19%**.
- Recall improved from **87.72% to 88.00%**.
- F1 Score improved from **85.10% to 86.06%**.
- BiLSTM used more parameters (**1,382,977 vs 1,331,521**).
- In the recorded training run, BiLSTM took **148.56 seconds** compared with **88.45 seconds** for the Basic LSTM.

Overall, the BiLSTM gave better validation metrics in this final comparison, with higher model complexity and longer training time.

---

# Part 6 — Regularization

The baseline model showed overfitting, so two regularization techniques were used:

1. **Dropout**
2. **Early Stopping**

## 1. Dropout

A dropout rate of **0.30** was used.

Dropout randomly disables some activations during training. This reduces the model's dependence on particular neurons and helps improve generalization.

## 2. Early Stopping

Early Stopping monitored validation loss:

```python
monitor="val_loss"
patience=2
restore_best_weights=True
```

Training stopped when the validation loss stopped improving, and the best weights were restored.

## Regularized LSTM Result

The best model state was at **epoch 3**:

```text
Validation Accuracy: 87.02%
Validation Loss:     0.3131
```

Compared with the Basic LSTM validation accuracy:

```text
87.02% - 84.64% = 2.38 percentage points improvement
```

The regularized model therefore generalized better than the baseline model.

---

# Part 7 — Model Evaluation

The final Regularized LSTM was evaluated on the complete unseen test dataset.

## Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | **86.48%** |
| Precision | **85.92%** |
| Recall | **87.26%** |
| F1 Score | **86.59%** |
| Test Loss | **0.3214** |

## Confusion Matrix

```text
                 Predicted
              Negative Positive

Actual Negative   10713    1787
Actual Positive    1592   10908
```

Therefore:

```text
TN = 10,713
FP = 1,787
FN = 1,592
TP = 10,908
```

Total incorrect predictions:

```text
3,379
```

Error rate:

```text
13.52%
```

### What is a False Positive?

A false positive means the review is actually **Negative**, but the model predicts **Positive**.

In this project:

```text
FP = 1,787
```

### What is a False Negative?

A false negative means the review is actually **Positive**, but the model predicts **Negative**.

In this project:

```text
FN = 1,592
```

### Generalization

The best validation accuracy was **87.02%**, while test accuracy was **86.48%**. The difference is only about **0.54 percentage points**, indicating reasonable generalization.

---

# Part 8 — Real-World Prediction

A function was created:

```python
predict_sentiment(review)
```

It accepts a raw review and performs:

```text
Raw Text
   ↓
Tokenization
   ↓
Integer Encoding
   ↓
Unknown Word Handling
   ↓
Padding / Truncation
   ↓
LSTM Model
   ↓
Sentiment + Confidence
```

## Custom Review Results

| # | Review Type | Predicted Sentiment | Confidence |
|---:|---|---|---:|
| 1 | Clearly positive | Positive | **88.17%** |
| 2 | Clearly negative | Negative | **99.60%** |
| 3 | Positive | Positive | **93.43%** |
| 4 | Negative | Negative | **99.35%** |
| 5 | Short positive | Positive | **83.78%** |
| 6 | Short negative | Negative | **88.08%** |
| 7 | Negation | Positive ❌ | **61.81%** |
| 8 | Negation / positive meaning | Negative ❌ | **65.55%** |
| 9 | Mixed sentiment | Negative | **94.17%** |
| 10 | Long / mixed | Positive | **82.96%** |

### Examples

```text
"This movie was absolutely amazing and I loved every minute of it."
→ Positive, 88.17%

"This movie was terrible, boring, and a complete waste of time."
→ Negative, 99.60%

"Amazing movie!"
→ Positive, 83.78%

"Very boring."
→ Negative, 88.08%
```

## Negation Analysis

The model struggled with some negation cases, for example:

```text
"I don't think this movie was good."
→ Positive ❌

"This was not a bad movie. I actually enjoyed it."
→ Negative ❌

"I did not enjoy this movie."
→ Positive ❌
```

This shows that the model can recognize strong sentiment words but does not always correctly understand how negation changes their meaning.

The 10 custom reviews are a qualitative test and are not a formal accuracy measurement.

---

# Part 9 — Error Analysis

The final test evaluation produced:

```text
Test reviews:        25,000
Incorrect predictions: 3,379
Error rate:           13.52%
```

The error analysis focused on the required categories: sarcasm, negation, short reviews, mixed sentiment, unusual vocabulary, and long reviews.

## Ten Incorrectly Classified Examples

| # | Actual | Predicted | Confidence | Length | Error Pattern |
|---:|---|---|---:|---:|---|
| 1 | Positive | Negative | 70.56% | 603 | Long, detailed review with many unknown tokens |
| 2 | Negative | Positive | 53.92% | 181 | Mixed sentiment and contradictory opinions |
| 3 | Negative | Positive | 91.12% | 134 | Positive words inside an overall negative review |
| 4 | Positive | Negative | 72.22% | 133 | Negation and conditional wording |
| 5 | Positive | Negative | 86.08% | 256 | Nuanced language and unusual vocabulary |
| 6 | Positive | Negative | 68.58% | 101 | Positive meaning expressed through complex wording |
| 7 | Negative | Positive | 95.94% | 324 | Sarcasm/irony with repeated negative words |
| 8 | Positive | Negative | 84.34% | 157 | Negative words inside an overall positive review |
| 9 | Positive | Negative | 91.56% | 135 | Negative-looking beginning followed by positive opinion |
| 10 | Positive | Negative | 94.25% | 173 | Negation/comparison and mixed language |

The full reviews are displayed in the notebook's Part 9 error-analysis output.

## Main Error Patterns

### Sarcasm

A strongly sarcastic negative review was predicted as positive with **95.94% confidence**. This shows that high confidence does not guarantee a correct prediction.

### Negation

Words such as `not`, `don't`, and `didn't` can change sentiment, but the model sometimes focused more on the sentiment word itself.

### Very Short Reviews

Short reviews contain less context, making them harder to classify reliably.

### Mixed Sentiment

Some reviews praise one part of a movie and criticize another part. This creates competing sentiment signals.

### Unusual Vocabulary

Words outside the selected vocabulary become `<UNK>`, which can remove useful information.

### Long Reviews

Long reviews can contain multiple opinions and changes in sentiment, making the overall classification more difficult.

### Negation-Word Check

The analysis also found **2,920 incorrect reviews containing one of the selected negation-related words**. This count only identifies the presence of such words; it does **not** mean that all 2,920 errors were caused by negation.

## Overall Observation

The model handles direct sentiment reasonably well but struggles more with sarcasm, negation, mixed sentiment, complex wording, very short or very long reviews, and unusual vocabulary.

---

# Part 10 — Bidirectional Sequence Modeling

## What is the difference between Standard LSTM and BiLSTM?

A standard LSTM processes the sequence in one direction:

```text
Word 1 → Word 2 → Word 3 → ... → Word N
```

A Bidirectional LSTM uses two LSTMs:

```text
Forward:
Word 1 → Word 2 → Word 3 → ... → Word N

Backward:
Word N → ... → Word 3 → Word 2 → Word 1
```

The two outputs are combined.

## Why can both directions help sentiment classification?

The meaning of a word can depend on information before and after it.

For example:

```text
"This movie was not good."
```

The word `good` is positive by itself, but `not` changes the meaning.

A BiLSTM can use contextual information from both sides of the sequence.

## Output Dimensions

The model used:

```text
Forward LSTM = 64 units
Backward LSTM = 64 units
```

Therefore:

```text
64 + 64 = 128
```

Observed output shapes:

```text
Input:
(1, 500)

Embedding:
(1, 500, 128)

BiLSTM:
(1, 128)
```

## Final BiLSTM Demonstration

The saved BiLSTM model was loaded and tested successfully.

```text
Input shape:                 (1, 500)
Embedding output shape:      (1, 500, 128)
BiLSTM output shape:         (1, 128)
Prediction probability:      0.9999988
Predicted sentiment:         Positive
```

## Comparison Result

The BiLSTM was already trained and compared with the Basic LSTM in Part 5.

```text
Basic LSTM validation accuracy: 84.64%
BiLSTM validation accuracy:     83.94%
```

For this particular experiment, the BiLSTM did not improve validation accuracy.

---

# Final Results

The **Regularized LSTM** was selected as the final model because regularization improved validation performance over the baseline.

| Metric | Final Result |
|---|---:|
| Test Accuracy | **86.48%** |
| Precision | **85.92%** |
| Recall | **87.26%** |
| F1 Score | **86.59%** |
| Test Loss | **0.3214** |

The final model showed good performance on direct sentiment patterns and reasonable generalization, while the error analysis identified areas that remain challenging, especially sarcasm, negation, mixed sentiment, and unusual vocabulary.

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Google Colab
- Google Drive

---

# Project Structure

```text
IMDB-Sentiment-Classification-LSTM/
│
├── IMDB_Sentiment_LSTM.ipynb
├── README.md
└── requirements.txt
```

The notebook contains the complete implementation and results for Parts 1–10.

---

# Conclusion

The project implemented a complete sentiment-classification pipeline using LSTM-based deep learning.

The workflow was:

```text
Dataset Analysis
      ↓
Text Preprocessing
      ↓
Embedding
      ↓
LSTM
      ↓
Training
      ↓
Model Comparison
      ↓
Regularization
      ↓
Test Evaluation
      ↓
Custom Predictions
      ↓
Error Analysis
      ↓
Bidirectional Modeling
```

The final Regularized LSTM achieved **86.48% test accuracy** and **86.59% F1 score**.

The experiments also showed that regularization improved generalization, while the more complex BiLSTM architecture did not necessarily produce better validation performance for this particular setup.
