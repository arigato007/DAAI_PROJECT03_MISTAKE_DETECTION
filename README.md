# Procedural Mistake Detection on CaptainCook4D

**Authors:** Arienzo Davide, Zito Simone, Bucaria Stefano  
**Course:** Data Analysis and Artificial Intelligence, Politecnico di Torino  
**Academic Year:** 2025/2026

---

## Abstract
This project addresses the challenge of **Procedural Mistake Detection** in egocentric videos using the **CaptainCook4D** benchmark. We evaluate standard baselines (SlowFast, Omnivore) and introduce a **LSTM** variant that achieves state-of-the-art performance on the challenging *Recordings* split. We also explore the **Perception Encoder** as a feature extractor to leverage vision-language alignment.

---

## Overview & Motivation

AI assistants in Augmented Reality (AR) and Robotics must do more than simply recognize actions; they must identify **when a procedure goes wrong**. 

While standard Action Recognition focuses on *what* is happening (e.g., "pouring water"), **Mistake Detection** requires understanding *how* it is happening and whether it violates the logical constraints of the recipe (e.g., "pouring water **too early**" or "**skipping** the mixing step").

**Key Challenges:**
* **Subtle Deviations:** Distinguishing between a correct action and a "Technique Error" often relies on fine-grained visual cues.
* **Temporal Dependencies:** Many errors are context-dependent (e.g., a step is only wrong if the previous one wasn't completed).
* **Domain Generalization:** Models often fail when tested in unseen kitchens with different lighting and actors (the *Recordings Split* problem).

**Our Goal:**
In this project, we move beyond simple frame-level classification. We benchmark state-of-the-art visual backbones (**Omnivore**, **SlowFast**) and demonstrate that a lightweight, causality-preserving **LSTM** aggregator generalizes better to unseen environments than complex Transformers for this specific task.

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
