# LISA 2025 Challenge – Team UPF

Complete implementation for the **[MICCAI 2025 LISA Challenge](https://www.synapse.org/Synapse:syn65670170/wiki/631438)**.

This repository contains the full solution developed by **Team UPF** for the **Low-field Pediatric Brain MRI Segmentation and Quality Assurance (LISA) Challenge**, held as part of **MICCAI 2025**.

## Challenge Overview

The **LISA Challenge** focuses on developing robust, accessible deep learning tools for **ultra–low-field (uLF) pediatric brain MRI**. These low-field scanners (e.g., 0.064T Hyperfine SWOOP) offer portable and cost-effective imaging solutions for children, especially in low- and middle-income countries where high-field MRI is limited.

The challenge targets two main objectives:
1. **Task 1 – Quality Assurance:** Quantify and classify MRI image quality and artifact severity.  
2. **Task 2 – Segmentation:** Automatically delineate key subcortical structures — the **hippocampi** (Task 2a) and **basal ganglia** (Task 2b).

The overarching goal is to advance **reliable and open deep learning pipelines** for pediatric neuroimaging in low-resource settings.

---

## 🧩 Our Approach

We participated in all competition tasks:

- **Task 1 – Quality Assurance (QA):** automatic evaluation of MRI image quality  
- **Task 2a – Hippocampus Segmentation:** delineation of hippocampal structures  
- **Task 2b – Basal Ganglia Segmentation:** segmentation of subcortical basal ganglia nuclei  

Our models were ranked:
- 🥉 **3rd place** on **Task 1 (Quality Assurance)**
- 🥈 **2nd place** on **Task 2b (Basal Ganglia Segmentation)**  
  
**Abstract:**  
Ultra–low-field (uLF) MRI systems offer portable and affordable neuroimaging solutions for pediatric patients, yet their low contrast and susceptibility to artifacts hinder accurate brain segmentation.  

This study addresses two critical challenges in uLF MRI: **automated quality assessment (QA)** and **anatomical structure segmentation**.

- **Quality Assurance (Task 1):**  
  We propose a **multi-label ordinal model** that integrates the ordered nature of artifact severity using an **ordinal loss**, and captures **artifact co-occurrence** via a **Bayesian Network**. The model is strengthened with **synthetic data augmentation** and **ensemble learning**, achieving a **composite accuracy of 0.84** across seven artifact categories.

- **Segmentation (Tasks 2a & 2b):**  
  We benchmark a **task-specific model ([nnU-Net](https://github.com/MIC-DKFZ/nnUNet))** trained from scratch against a **foundation model (SAM-Med3D)** fine-tuned for uLF data.  
  nnU-Net achieved **mean Dice scores of 0.72** for hippocampi and **0.86** for basal ganglia. Lightweight fine-tuning of SAM-Med3D reached a comparable **0.70 Dice** for hippocampi, demonstrating the potential of **foundation models** for medical imaging under domain shift.
  The  framework

These results underscore the promise of combining robust segmentation frameworks with foundation models for **low-field MRI** in **resource-limited environments**.
---
