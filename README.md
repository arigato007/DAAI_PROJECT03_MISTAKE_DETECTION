# Procedural Mistake Detection on CaptainCook4D

**Authors:** Arienzo Davide, Zito Simone, Bucaria Stefano  
**Course:** Data Analysis and Artificial Intelligence, Politecnico di Torino  
**Academic Year:** 2025/2026

---

## Abstract
This project investigates procedural mistake detection in complex human activities, with a focus on instructional cooking scenarios. The goal is to move beyond standard action recognition by identifying subtle semantic and temporal errors that occur during task execution. We evaluate multiple architectural variants and visual backbones, reproduce established baselines, and introduce temporal modeling strategies to improve robustness. In particular, we show that explicitly modeling sequential dependencies significantly enhances mistake detection performance, especially under challenging generalization settings.

---

## Overview & Motivation

Procedural understanding is a key component of intelligent systems designed to assist humans in real-world tasks. In domains such as cooking, effective assistance requires not only recognizing which action is being performed, but also detecting when a mistake occurs, such as skipping a step, performing an action in the wrong order, or committing a measurement or timing error.

This project is based on the CaptainCook4D benchmark and focuses on the task of procedural mistake detection from egocentric video data. We analyze the problem under different evaluation protocols, including Step and Recordings splits, which test both intra-procedure consistency and generalization to unseen environments. The repository contains the full pipeline for feature extraction, model training, and evaluation, enabling reproducibility.

---

## Data & Checkpoints
Due to GitHub storage limits, the pre-extracted features and trained model checkpoints are hosted externally.

**[CLICK HERE TO DOWNLOAD DATA & CHECKPOINTS (Google Drive)](https://drive.google.com/drive/folders/1Ne50XFNHZEsQ3K0_cYI4JdBWqJm3_B7h?usp=drive_link)**

**Setup:**
The folder structure on Google Drive **already matches** the project requirements.
1. Download the content from the Drive link.
2. Extract/Place the folders directly into the project root so that you have:

* `captain_cook_4d_gopro_resized/` (Contains raw videos for Perception Encoder)
* `features/` (Contains pre-extracted features for Omnivore & SlowFast)
* `error_recognition/checkpoints/` (Contains the trained `.pth` models)

---

## Methodology Approach
This project follows a progressive experimental pipeline for procedural mistake detection.

First, we reproduce the official CaptainCook4D baselines by performing mistake detection using pre-extracted Omnivore and SlowFast features, combined with MLP and Transformer variants. Experiments are conducted under both Step and Recordings splits to evaluate performance and generalization.

Second, we extend the baseline framework by introducing an LSTM-based model, explicitly designed to capture long-term temporal dependencies in procedural activities. The LSTM is trained on the same fixed Omnivore and SlowFast features and evaluated using the same protocols for fair comparison.

Finally, we integrate a new feature extraction backbone based on a Perception Encoder. Features are extracted directly from raw videos and used to train all three variants (MLP, Transformer, and LSTM). Results are compared to analyze the impact of richer visual representations and temporal modeling on mistake detection performance.
