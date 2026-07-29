# Multimodal Analysis of Voice, Gait and Handwriting Signals for Parkinson Disease Detection

## Overview

This project presents a multimodal deep learning framework for Parkinson's disease detection by integrating handwriting images, gait signals, and voice recordings. Individual models are trained for each modality and their prediction probabilities are combined using decision level (late) fusion to improve diagnostic performance. Explainable AI techniques are incorporated to provide model interpretability.

## Features

* Handwriting based Parkinson's disease detection using DenseNet121
* Gait analysis using CNN BiLSTM architecture
* Voice analysis using Ensemble Learning
* Decision level multimodal fusion
* Explainable AI using Grad CAM and PCA based feature interpretation
* Graphical User Interface for multimodal prediction

---

## Project Architecture

### Overall System Architecture

![System Architecture](images/system_architecture.png)

### Decision Level Fusion

![Fusion](images/fusion_explanation.png)

---

## Individual Models

### Handwriting Model

![Handwriting Architecture](images/handwriting_architecture.png)

### Gait Model

![Gait Architecture](images/gait_architecture.png)

### Voice Model

![Voice Architecture](images/voice_architecture.png)

---

## Explainable AI

### Grad CAM Visualization

![GradCAM Spiral](images/gradcam_spiral.jpg)

![GradCAM Wave](images/gradcam_wave.jpg)

### Voice Feature Importance

![PCA](images/voice_pca_importance.png)

### Temporal Force Pattern

![Gait](images/gait_temporal_force.png)

---

## Graphical User Interface

![GUI](images/application_interface.png)

---

## Technologies Used

* Python
* TensorFlow
* Keras
* Scikit Learn
* OpenCV
* NumPy
* Pandas
* Matplotlib
* SHAP

---

## Datasets

### Handwriting

Parkinson's Handwriting Dataset

### Gait

PhysioNet Gait in Parkinson Disease Dataset

### Voice

Voice Samples for Patients with Parkinson's Disease and Healthy Controls

---

## Repository Structure

```
Parkinson-Disease-Detection-Multimodal/
│
├── app.py
├── fusion.py
├── pd_gait.ipynb
├── pd_handwriting.ipynb
├── pd_voice.ipynb
├── images/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/amulyajayaram/Parkinson-Disease-Detection-Multimodal.git
```

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application:

```bash
python app.py
```

---

## Results

The proposed multimodal framework combines predictions from handwriting, gait, and voice modalities using decision level fusion to improve Parkinson's disease detection performance while providing explainable predictions through visualization techniques.

---

## Conference Publication

This work has been accepted for presentation at **ICACKE 2026**.

**Paper Title**

**Multimodal Analysis of Voice, Gait and Handwriting Signals for Parkinson Disease Detection**

---

## License

This project is licensed under the MIT License.
