# 🚀 Speech Robustness Framework

### *Towards Fair, Private, and Robust Speech AI Systems*

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge\&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge\&logo=huggingface)
![Speech](https://img.shields.io/badge/Domain-Speech%20AI-green?style=for-the-badge)
![Research](https://img.shields.io/badge/Type-Research%20Project-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

---

## 🌟 TL;DR

> A **research-grade speech AI framework** that improves:

* 🎯 **Robustness** → Works in noisy environments
* ⚖️ **Fairness** → Reduces demographic bias
* 🔐 **Privacy** → Hides speaker identity

📈 Achieves **97%+ accuracy in noisy conditions** with fairness-aware training and privacy-preserving transformations.

---

## 🧠 Project Motivation

Most speech AI systems today:

❌ Overfit to clean environments
❌ Ignore demographic bias
❌ Leak speaker identity

This project addresses all three — **simultaneously**.

---

## 🏗️ System Architecture

```
Raw Audio
   │
   ├── Feature Extraction (MFCC - Manual)
   │
   ├── Segmentation (Voiced / Unvoiced)
   │
   ├── Alignment (Wav2Vec2)
   │
   ├── Representation Learning
   │      ├── Disentanglement (GRL)
   │      └── Temporal Attention  ⭐ (Proposed)
   │
   ├── Fairness Module
   │
   ├── Privacy Module (Voice Obfuscation)
   │
   └── Evaluation (RMSE, LSD, Accuracy)
```

---

## 🚀 Key Innovations

### 🔹 Manual MFCC Pipeline (From Scratch)

* Full DSP implementation (no librosa shortcuts)
* Transparent feature engineering

📄 Based on: 

---

### 🔹 Cepstrum-Based Boundary Detection

* Uses quefrency domain separation
* Detects voiced/unvoiced segments automatically

---

### 🔹 Spectral Leakage Analysis

| Window      | SNR (dB) | Leakage   |
| ----------- | -------- | --------- |
| Rectangular | 15.91    | High      |
| Hamming     | 13.68    | Optimal ✅ |
| Hanning     | 13.50    | Smooth    |

✔ Shows why **Hamming window dominates speech processing** 

---

### 🔹 Transformer-Based Alignment

* Wav2Vec2 forced alignment
* RMSE < **30 ms**

---

### 🔹 🧠 Disentangled Representation Learning

* Removes environmental noise from embeddings
* Uses **Gradient Reversal Layer (GRL)**

📄 Refer: 

---

### 🔹 🚀 Novel Contribution: Temporal Attention

❌ Problem: Global pooling dilutes useful speech
✅ Solution: Learn frame importance dynamically

```python
Weighted_Features = Σ (Features × Softmax(Attention))
```

📊 Results:

| Model                  | Clean    | Noisy     |
| ---------------------- | -------- | --------- |
| Baseline               | 77.5%    | 77.1%     |
| Disentangled           | 87.5%    | 80.0%     |
| **+ Attention (Ours)** | **100%** | **97.1%** |

✔ Massive robustness gain 

---

### 🔹 ⚖️ Ethical Auditing (Bias Detection)

* Found **representation imbalance**
* WER gap:

  * Standard: ~0.06–0.08
  * Atypical: ~0.18–0.22

🚨 Confirms **real-world bias in datasets** 

---

### 🔹 Fairness-Aware Loss

```math
Total Loss = ASR Loss + λ × |Loss_group1 - Loss_group2|
```

✔ Forces equal performance across demographics

---

### 🔹 🔐 Privacy-Preserving Speech

* Pitch-based voice transformation:

  * Male → Female
  * Old → Young

✔ Identity hidden, speech meaning preserved 

---

### 🔹 📊 Quality Validation

* Metric: **Log-Spectral Distance (LSD)**
* Scores: ~5.4–5.8 dB

✔ No perceptual degradation 

---

## ⚡ Quick Start

```bash
git clone https://github.com/Astro-Phile/Speech-Robustness-Framework.git
cd Speech-Robustness-Framework
pip install -r requirements.txt
```

---

## ▶️ Run Everything

```bash
# Feature Extraction
python src/mfcc_manual.py

# Spectral Analysis
python src/leakage_snr.py

# Segmentation
python src/voiced_unvoiced.py

# Alignment
python src/phonetic_mapping.py

# Training
python src/train.py
```

---

## 📊 Results Dashboard

| Metric              | Value      |
| ------------------- | ---------- |
| Alignment RMSE      | < 30 ms    |
| Noisy Accuracy      | **97.14%** |
| Bias Gap (Before)   | High       |
| Bias Gap (After)    | Reduced    |
| Privacy Score (LSD) | ~5.5 dB    |

---

## 🧪 Tech Stack

* 🐍 Python
* 🔥 PyTorch
* 🤗 Transformers (Wav2Vec2)
* 📊 NumPy, Matplotlib
* 🎧 DSP (FFT, Cepstrum, MFCC)

---

## 📁 Project Structure

```
Speech-Robustness-Framework/
│
├── src/
├── q2/
├── q3/
├── data/
├── requirements.txt
└── README.md
```

---

## 🎯 Real-World Applications

* 🎙️ Voice Assistants (robust to noise)
* 🏥 Healthcare Speech Systems
* 🛡️ Privacy-Preserving Voice Tech
* 🌍 Fair AI for diverse populations

---

## 🔮 Future Work

* Real-time anonymization
* Multilingual bias auditing
* Edge deployment
* Self-supervised fairness learning

---

## 👨‍💻 Author

**Aditya Kashyap**
🔗 [https://github.com/Astro-Phile](https://github.com/Astro-Phile)

---

## 📜 License

MIT License

---

## ⭐ Support

If this project helped you:

👉 Star ⭐ the repo
👉 Share with ML researchers
👉 Use it in your projects

---

## 💡 Final Note

> Building accurate AI is easy.
> Building **fair, private, and robust AI** is the real challenge.

---
