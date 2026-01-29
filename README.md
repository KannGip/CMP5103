# CMP5103

# Improving Cyberbullying Detection Using BERT and Paraphrase Augmentation

**Author:** Kanan Nasibli  
**Affiliation:** Computer Engineering, Bahçeşehir University  
**Contact:** [kanan.nasibli@bahcesehir.edu.tr](mailto:kanan.nasibli@bahcesehir.edu.tr) | Student ID: 2105290

## Project Overview

Cyberbullying remains a serious problem on social media platforms like X (Twitter), Instagram, and online forums. This project develops a **multi-label toxic comment classifier** based on **BERT**, enhanced with **Large Language Model (LLM) paraphrase augmentation** to better detect rare and severe toxic categories such as **threat** and **identity_hate**.

The approach focuses on addressing **severe class imbalance** in real-world toxicity datasets through targeted synthetic data generation.

## Key Features

- **Transformer Architecture** — Fine-tuned `bert-base-uncased` with subword tokenization (WordPiece) to robustly handle intentional misspellings, slang, and noisy cyberbullying text.
- **Generative Augmentation** — Uses the **OpenAI API** to create high-quality paraphrases specifically for underrepresented toxic labels, improving model generalization.
- **Data-Centric Optimization** — Includes a preprocessing pipeline that removes non-informative elements (URLs, user mentions, excessive punctuation) while preserving toxic intent.

## Methodology

### 1. Dataset
- **Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (~159,571 labeled comments)
- **Labels:** multi-label (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Challenge:** Severe class imbalance  
  - threat: ~478 samples  
  - identity_hate: ~1,405 samples  
  - toxic: ~15,294 samples (majority class)

### 2. Augmentation Strategy
- **Target classes:** Primarily **threat** and **identity_hate**
- **Technique:** Generate **two unique paraphrases** per minority sample using OpenAI's language model
- **Goal:** Increase representation of rare classes → force the model to learn more robust boundaries for subtle/hard toxic cases

### 3. Training and Optimization
- **Hardware:** NVIDIA A100 GPUs (via Google Colab Pro or equivalent)
- **Optimizer:** AdamW
- **Learning rate:** 1e-5 (refined for stability)
- **Epochs:** Early stopping applied — training typically peaks at 2 epochs (performance drops in epoch 3 due to overfitting on augmented data)

## Results

The augmented model shows consistent gains, especially in **Macro F1-score** (which better reflects performance on rare classes).

| Configuration              | Macro F1 (%) | Micro F1 (%) |
|----------------------------|--------------|--------------|
| Baseline BERT              | 65.23        | 78.35        |
| Augmented BERT             | 67.42        | 78.90        |

**Note:** Macro F1 is the primary metric here, as it gives equal weight to all classes and highlights improvements on minority toxic labels (threat, identity_hate).

## Future Improvements

- **Enhanced Loss Functions** — Experiment with Focal Loss or optimized class weighting to further focus on hard examples
- **Multi-Dataset Training** — Combine Jigsaw with other toxicity/cyberbullying datasets (e.g., from different platforms/languages) for better generalization
- **API & Efficiency** — Switch to faster/more recent LLMs (e.g., newer OpenAI models or open-source alternatives) to reduce paraphrase generation time and cost

## Requirements

```bash
pip install transformers torch pandas scikit-learn openai
