## 🚀 Getting Started

### 1. Prerequisites
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/CaptainCook-Mistake-Detection.git](https://github.com/YOUR_USERNAME/CaptainCook-Mistake-Detection.git)
cd CaptainCook-Mistake-Detection
pip install -r requirements.txt

# Procedural Mistake Detection on CaptainCook4D 👨‍🍳🚫

**Authors:** Arienzo Davide, Zito Simone, Bucaria Stefano  
**Course:** [Inserire Nome del Corso], Politecnico di Torino  
**Academic Year:** 2024/2025

---

## 📝 Abstract

This project addresses the challenge of **Procedural Mistake Detection** in egocentric videos using the **CaptainCook4D** benchmark. We evaluate standard baselines and propose architectural improvements to enhance error recognition in unseen environments.

Our contributions are threefold:
1.  **Baseline Analysis:** We benchmark **SlowFast** and **Omnivore** backbones with **MLP** and **Transformer** heads.
2.  **Sequential Modeling (Ours):** We introduce a **Unidirectional LSTM** variant that achieves state-of-the-art performance on the challenging *Recordings* split, effectively capturing temporal dependencies.
3.  **Visual Representation (Ours):** We explore the **Perception Encoder** as a feature extractor to leverage vision-language alignment for detecting fine-grained technique errors.

---

## 📊 Key Results

We evaluated our models on two splits: **Step** (familiar environment) and **Recordings** (unseen environment).

| Split | Backbone | Variant | F1-Score | Accuracy | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | Omnivore | Transformer | 55.39% | 69.92% | 0.756 |
| **Step** | Omnivore | **LSTM (Ours)** | **55.56%** | **74.94%** | **0.791** |
| | | | | | |
| **Recordings** | Omnivore | Transformer | 40.73% | 61.40% | 0.622 |
| **Recordings** | Omnivore | **LSTM (Ours)** | **46.92%** | **58.87%** | **0.606** |

> **Key Insight:** The LSTM variant proves to be more robust than Transformers in generalizing to unseen actors and kitchens, offering the best trade-off between Precision and Recall.

---

## 📂 Project Structure

```text
Mistake_Detection_Project/
├── configs/              # Configuration files (YAML)
├── data/                 # Dataset splits and labels
├── models/
│   ├── mlp.py            # MLP Baseline
│   ├── transformer.py    # Transformer Baseline
│   └── lstm.py           # LSTM Implementation (Ours)
├── utils/                # Helper functions (loading, metrics)
├── train.py              # Main training script
├── evaluate.py           # Inference and metric calculation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
