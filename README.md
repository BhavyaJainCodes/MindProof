# 🧠 MindProof — Argument Intelligence Lab

### A Hybrid Machine Learning Framework for Argument Quality Assessment and Stance Classification

MindProof is a machine learning project that analyzes written arguments using natural language processing (NLP), semantic embeddings, and handcrafted linguistic features.

The system assesses argument quality, predicts stance, and provides additional insights to help users understand and improve their arguments.

MindProof is designed to support critical thinking and argument analysis. It does not determine which person is right or whether an argument is factually true.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Research and Academic Context](#research-and-academic-context)
- [Authors](#authors)

---

## 📖 Overview

Arguments are an important part of discussions, debates, and decision-making. However, analyzing the structure and quality of written arguments can be challenging.

MindProof explores how machine learning and NLP can assist with argument analysis.

The framework combines multiple types of features to analyze textual arguments:

- **TF-IDF features** to represent the importance of words and phrases.
- **Semantic embeddings** to represent aspects of textual meaning.
- **Handcrafted linguistic features** to capture measurable characteristics of arguments.
- **Stance classification** to predict whether an argument supports or opposes a topic.

The project also includes rule-based indicators that highlight possible reasoning fallacies and provides feedback intended to help users improve their arguments.

---

## ✨ Key Features

### 1. Argument Quality Assessment
Predicts an argument quality score based on patterns learned from annotated argument data.

### 2. Stance Classification
Predicts whether an argument is in favor of or against a topic.

### 3. Hybrid Feature Engineering
Combines traditional text representations, semantic embeddings, and handcrafted linguistic features.

### 4. Argument Analysis
Provides additional information about the structure and characteristics of an argument.

### 5. Rule-Based Fallacy Indicators
Identifies selected linguistic patterns that may indicate possible reasoning fallacies.

These indicators are preliminary signals and should not be treated as definitive judgments.

### 6. Improvement Suggestions
Provides feedback intended to help users make their arguments clearer and better supported.

### 7. Model Evaluation
Compares multiple model versions using regression and classification metrics.

### 8. Desktop User Interface
Provides an interface for entering arguments and viewing analysis results.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and feature extraction |
| TF-IDF | Text vectorization |
| Sentence Transformers | Semantic embeddings |
| CustomTkinter | Desktop user interface |
| Matplotlib | Data visualization |
| ReportLab | PDF generation, if enabled in the application |

---

## 🏗️ System Architecture

The project follows a machine learning pipeline:

1. **Input:** The user provides a topic and written arguments.
2. **Preprocessing:** Text is cleaned and prepared for analysis.
3. **Feature Extraction:** TF-IDF, semantic embeddings, and handcrafted linguistic features are generated.
4. **Quality Assessment:** A regression model predicts the argument quality score.
5. **Stance Classification:** A classification model predicts the argument's stance toward the topic.
6. **Additional Analysis:** The application generates supplementary insights and feedback.
7. **Output:** The user receives the predictions and analysis results.

---

## 📚 Dataset

MindProof uses the IBM Research Argument Quality Ranking 30K dataset.

**Dataset source:**

https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k

The dataset contains arguments and associated annotation information used in argument quality research.

The recorded local dataset split sizes are:

| Split | Number of Examples |
|---|---:|
| Training | 20,974 |
| Development | 3,208 |
| Test | 6,315 |
| **Total** | **30,497** |

The dataset is used to train and evaluate the project's machine learning models.

Please consult the original dataset documentation for its licensing, annotation methodology, and permitted uses.

---

## 🔍 Feature Engineering

MindProof explores three categories of features.

### 1. TF-IDF Features

Term Frequency–Inverse Document Frequency represents the importance of words and phrases in textual documents.

It helps the model identify informative textual patterns.

### 2. Semantic Embeddings

The project uses the `sentence-transformers/all-MiniLM-L6-v2` model to generate 384-dimensional semantic embeddings.

These embeddings provide numerical representations of textual meaning.

### 3. Handcrafted Linguistic Features

The project extracts measurable characteristics of arguments, including:

- Word count
- Unique word count and ratio
- Sentence count
- Average sentence length
- Reasoning-related markers
- Evidence-related markers
- Example-related markers
- Contrast-related markers
- Conclusion-related markers
- Question and exclamation counts
- Number counts
- Multi-sentence indicators

These features complement TF-IDF and semantic representations.

---

## 🤖 Machine Learning Models

The project explores multiple model versions.

| Version | Approach |
|---|---|
| V2 | Traditional text features and handcrafted linguistic features |
| V3 | Semantic transformer embeddings and a hybrid stance classification pipeline |
| V4 | Hybrid representation combining TF-IDF, semantic embeddings, and handcrafted features |

The models are evaluated using separate quality assessment and stance classification tasks.

The best-performing version depends on the evaluation metric. No single version should be considered best across every metric without further evidence.

---

## 📊 Model Evaluation

The following table presents the recorded test results.

### Argument Quality Assessment

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| V2 | 0.1459 | 0.1847 | 0.1383 |
| V3 | 0.1529 | 0.1932 | 0.0563 |
| V4 | 0.1493 | 0.1848 | 0.1371 |

**Metrics:**

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

Lower MAE and RMSE indicate smaller prediction errors. Higher R² indicates that the model explains more variation in the target values.

The relatively low R² values indicate that predicting argument quality remains challenging.

### Stance Classification

| Model | Accuracy | F1-score |
|---|---:|---:|
| V2 | 66.48% | 66.47% |
| V3 | 57.89% | 56.20% |
| V4 | 66.48% | 66.47% |

These results reflect the recorded test evaluations and should be interpreted in the context of the dataset, annotation process, and model limitations.

---

## 📁 Project Structure

The repository contains the following major components:

```text
MindProof/
│
├── analysis/
├── datasets/
│   └── argument_quality/
├── models/
├── reports/
├── research/
│
├── app.py
├── app_ui.py
├── model_features.py
├── predict_v4.py
├── train_models_v4.py
│
├── requirements.txt
└── README.md
