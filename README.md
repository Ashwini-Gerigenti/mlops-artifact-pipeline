# ML Ops Assignment 2 – Digit Classification Pipeline

## 👨‍💻 Team Members
- Ashwini Gerigenti: g24ai1082@iitj.ac.in  
---

## 📌 Objective

This project implements a CI/CD pipeline for a digit classification model using **Logistic Regression** on the built-in `digits` dataset from `sklearn`. It focuses on modular ML pipeline design, automation using GitHub Actions, and artifact management using GitHub workflows.

---

## 🚀 Project Outcomes

- 📦 Model training using parameters from a JSON config
- ✅ Unit testing with `pytest`
- 🤖 Multi-step inference workflow with CI/CD on GitHub Actions
- 🔗 Artifacts (trained model) passed between jobs
- 📊 Performance reporting (accuracy, loss)

---

## 🏗️ Project Structure

.
├── config/

│ └── config.json # Hyperparameters

├── src/

│ ├── train.py # Train model

│ ├── inference.py # Predict using saved model

│ └── utils.py # Utility functions (if any)

├── tests/

│ └── test_train.py # Unit tests for training pipeline

├── .github/

│ └── workflows/

│ ├── train.yml # CI for training

│ ├── test.yml # CI for testing

│ └── inference.yml # CI for inference with artifact passing

├── requirements.txt

└── README.md
