# Radar Target Classification System

An end-to-end machine learning system for classifying airborne radar targets — Bird, Drone, Aircraft, and Stealth UAV — using engineered micro-Doppler signal features. The project investigates radar target classification using a physics-inspired synthetic dataset and evaluates model robustness under signal perturbations.

**Live Demo:** https://radar-target-classification.streamlit.app/

---

# Problem

Modern airspace monitoring systems must distinguish between harmless objects, civilian drones, conventional aircraft, and stealth threats using radar returns. Access to real radar datasets is often restricted, making experimentation difficult.

This project uses a physics-inspired synthetic micro-Doppler dataset that simulates characteristic signal behaviour of different airborne targets and explores how machine learning models can classify them using interpretable signal features.

---

# Dataset

**Source:** Kaggle Micro-Doppler Aerial Classification Dataset

Dataset Characteristics:

* 2,800 samples
* 700 samples per class
* 4 classes:

  * Bird
  * Drone
  * Aircraft
  * Stealth UAV
* 100 timesteps per sample
* 3 signals per timestep:

  * Amplitude
  * Velocity
  * Energy

Raw representation:

100 timesteps × 3 signals = 300 values per sample

The dataset is synthetically generated using class-specific signal generation mechanisms inspired by micro-Doppler behaviour.

---

# What This System Does

* Extracts meaningful radar signal features from raw time-series data
* Classifies airborne targets using multiple machine learning models
* Compares model performance across clean and noisy conditions
* Provides model interpretability through feature importance and SHAP explanations
* Deploys an interactive Streamlit dashboard for live classification and signal exploration

---

# Feature Engineering

Instead of directly training on all 300 raw values, the radar signals are transformed into a compact set of physically meaningful features.

### Statistical Features

Extracted from amplitude and velocity signals:

* Mean
* Standard Deviation
* Variance
* Minimum
* Maximum
* Range
* Signal Energy

These features capture signal magnitude and variability.

### Frequency Feature

Dominant Frequency (FFT)

Captures periodic rotor signatures commonly associated with drones.

### Temporal Feature

Autocorrelation (Lag 5)

Measures signal self-similarity and periodic behaviour over time.

### Feature Selection

The original Energy signal was removed after analysis showed:

Energy = abs(Amplitude)

Correlation between the two signals was effectively 1.0, making the Energy channel mathematically redundant and adding no additional predictive information.

Final feature count:

16 engineered features

---

# Key Observation

One of the most important findings from this project was that a single engineered feature, amplitude variance, achieved approximately 99.5% classification accuracy by itself.

This indicates that the synthetic classes are highly separable and explains why multiple machine learning models achieve near-perfect performance.

This behaviour was verified through:

* Single-feature experiments
* Feature importance analysis
* Class-wise feature distribution analysis

Rather than blindly accepting perfect scores, the project investigates why the classification task is unusually easy.

---

# Model Comparison

The following models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Support Vector Machine (SVM)

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 100%     |
| Decision Tree       | 100%     |
| Random Forest       | 100%     |
| SVM                 | 100%     |
| XGBoost             | 99.8%    |

All models achieved near-perfect performance on the clean synthetic dataset due to strong class separability.

---

# Robustness Analysis

Because the clean dataset produced nearly identical performance across models, additional robustness testing was performed.

Gaussian noise was injected into the engineered feature space to evaluate classifier sensitivity to feature perturbations.

| Noise Level | Logistic Regression | Random Forest | XGBoost | SVM   |
| ----------- | ------------------- | ------------- | ------- | ----- |
| 0.0         | 100%                | 100%          | 99.8%   | 100%  |
| 0.1         | 100%                | 100%          | 99.6%   | 100%  |
| 0.3         | 100%                | 99.5%         | 96.1%   | 100%  |
| 0.5         | 99.1%               | 96.3%         | 87.5%   | 99.5% |

Key findings:

* SVM showed the strongest robustness under increasing feature perturbation.
* Random Forest maintained strong performance while remaining interpretable.
* XGBoost degraded most rapidly under heavy noise.
* Aircraft remained the easiest class to classify under noise.
* Drone classification degraded most under perturbation.

These results highlight the importance of robustness testing when benchmark accuracy alone cannot distinguish model quality.

---

# Feature Importance

Random Forest feature importance revealed that amplitude-derived features dominate classification performance.

Top features:

| Feature                      | Importance |
| ---------------------------- | ---------- |
| Amplitude Variance           | 15.3%      |
| Amplitude Standard Deviation | 14.9%      |
| Amplitude Energy             | 13.8%      |
| Amplitude Range              | 9.5%       |
| Velocity Variance            | 8.1%       |

Within this dataset:

* Amplitude features contribute approximately 67.4% of total importance
* Velocity features contribute approximately 32.6%

This indicates that amplitude-derived information provides most of the predictive power for the synthetic target classes.

---

# Model Deployment

Random Forest was selected as the deployment model.

Reason:

* Perfect performance on the clean dataset
* Feature importance analysis
* SHAP-based prediction explanations
* Better interpretability for demonstration purposes

Although SVM demonstrated superior robustness under feature perturbation, Random Forest provides significantly better explainability and insight into model behaviour.

---

# Streamlit Dashboard

### Live Classifier

* Random target simulation using real dataset samples
* Signal visualization
* Class prediction
* Confidence scores
* SHAP explanation plots

### Signal Explorer

* Compare signal patterns across target classes
* Visualize amplitude and velocity behaviour

### Robustness Analysis

* Noise sensitivity comparison
* Per-model performance degradation
* Interactive analysis of perturbation effects

### Model Intelligence

* Feature importance ranking
* Model comparison
* Classification insights

---

# Project Structure

```text
Radar-Target-Classification/
│
├── radar_streamlit.py
├── rf_model_rc.pkl
├── scaler_rc.pkl
├── feature_names_rc.json
├── radar_samples.json
├── radar_signals.json
├── requirements.txt
└── README.md
```

---

# Tech Stack

* Python
* NumPy
* pandas
* scikit-learn
* XGBoost
* SHAP
* matplotlib
* Streamlit

---

# How To Run Locally

```bash
git clone https://github.com/Samarth9192/Radar-Target-Classification.git

cd Radar-Target-Classification

pip install -r requirements.txt

streamlit run radar_streamlit.py
```

---

# Limitations

* Dataset is synthetically generated and does not represent the full complexity of real radar environments.
* Real radar systems encounter clutter, multipath effects, electronic interference, and overlapping signatures not present here.
* Robustness testing was performed using Gaussian perturbations in engineered feature space rather than true radar signal corruption.
* Amplitude variance alone achieves approximately 99.5% accuracy, indicating that the synthetic classes are significantly easier to separate than real-world radar targets.
* Results should therefore be interpreted as a machine learning study on synthetic radar-inspired signals rather than a production-ready radar classification system.

---

# Future Work

* Inject noise at the raw signal level before feature extraction
* Evaluate robustness against structured radar interference
* Explore sequence-based models such as LSTM and GRU
* Test on real micro-Doppler radar datasets when available
* Extend classification to additional airborne target categories
* Investigate domain adaptation from synthetic to real radar data

```
```
