# Procedural Mistake Detection on CaptainCook4D

**Authors:** Arienzo Davide, Zito Simone, Bucaria Stefano  
**Course:** Data Analysis and Artificial Intelligence, Politecnico di Torino  
**Academic Year:** 2025/2026

---

## Abstract
This project addresses the challenge of **Procedural Mistake Detection** in egocentric videos using the **CaptainCook4D** benchmark. We evaluate standard baselines (SlowFast, Omnivore) and introduce a **Unidirectional LSTM** variant that achieves state-of-the-art performance on the challenging *Recordings* split. We also explore the **Perception Encoder** as a feature extractor to leverage vision-language alignment.

---

## Data & Checkpoints
Due to GitHub storage limits, the pre-extracted features and trained model checkpoints are hosted externally.

**[CLICK HERE TO DOWNLOAD DATA & CHECKPOINTS (Google Drive)](https://drive.google.com/drive/folders/1Ne50XFNHZEsQ3K0_cYI4JdBWqJm3_B7h?usp=drive_link)**

**Setup:**
1. Download the zip folder from the link above.
2. Extract it inside the project root so that you have a `data/` and `checkpoints/` folder.

---

## 📊 Key Results

| Split | Backbone | Variant | F1-Score | Accuracy | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | Omnivore | Transformer | 55.39% | 69.92% | 0.756 |
| **Step** | Omnivore | **LSTM (Ours)** | **55.56%** | **74.94%** | **0.791** |
| | | | | | |
| **Recordings** | Omnivore | Transformer | 40.73% | 61.40% | 0.622 |
| **Recordings** | Omnivore | **LSTM (Ours)** | **46.92%** | **58.87%** | **0.606** |

> **Insight:** The LSTM variant outperforms the Transformer on the unseen Recordings split, offering a better trade-off between Precision and Recall.

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
