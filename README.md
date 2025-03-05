🏥 Breast Cancer Classification with Decision Trees & Random Forests
📌 Overview
This project explores breast cancer classification using Decision Tree and Random Forest models. The objective is to predict whether a tumor is malignant or benign based on various diagnostic features.

The project includes:

Data preprocessing and feature engineering.
Interactive Shiny app to visualize decision trees and random forests.
Model evaluation using confusion matrices, ROC curves, and feature importance analysis.
Performance comparison between Decision Tree and Random Forest classifiers.
📂 Project Setup
Data
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset.
Source: UCI Machine Learning Repository.
Size: 569 observations, 30 features.
Key Variables:
Input Features: Cell nucleus characteristics (radius, texture, perimeter, area, smoothness, etc.).
Target Variable: Diagnosis classified as Malignant (1) or Benign (0).
Tools & Technologies
Python for data processing, visualization, and modeling.
Libraries Used: pandas, matplotlib, seaborn, sklearn, shiny.
Machine Learning Models:
Decision Tree Classifier
Random Forest Classifier
🔍 Methodology
1️⃣ Data Preprocessing
Fetched dataset from UCI Machine Learning Repository.
Encoded target variable: M → 1 (Malignant), B → 0 (Benign).
Split dataset into training (85%) and test (15%) sets.
Applied Gini impurity and Entropy criteria for model evaluation.
2️⃣ Interactive Shiny App
Users can select model type (Decision Tree or Random Forest).
Various visualization options:
Decision Tree structure
Confusion Matrix
ROC Curve
Feature Importance
Error rate vs. Number of Trees (for Random Forest).
3️⃣ Model Training & Evaluation
📊 Decision Tree Classifier
Splitting Criteria: Gini impurity & Entropy.
Pruning: Applied cost-complexity pruning (ccp_alpha=0.01).
Performance (Test Set):
Gini: Accuracy 85.71%, AUC 0.89.
Entropy: Accuracy 88.57%, AUC 0.92.
📌 Random Forest Classifier
Hyperparameter: n_estimators=50.
Performance (Test Set):
Gini: Accuracy 90.00%, AUC 0.99.
Entropy: Accuracy 92.86%, AUC 0.99.
📈 Key Results
Model	Accuracy	Sensitivity	Specificity	AUC
Decision Tree (Gini)	85.71%	90.63%	80.56%	0.89
Decision Tree (Entropy)	88.57%	93.75%	83.33%	0.92
Random Forest (Gini)	90.00%	90.63%	87.88%	0.99
Random Forest (Entropy)	92.86%	90.63%	93.55%	0.99
📌 Final Model Selection: Random Forest (Entropy)
Highest accuracy and AUC (0.99).
Lower classification error and more balanced feature importance distribution.
More robust than a single Decision Tree but requires more computational power.
