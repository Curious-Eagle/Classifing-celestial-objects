# Classifing-celestial-objects

1. Overview

  - Goal: Automate the classification of celestial objects (GALAXY, QSO, STAR) based on spectroscopic data (e.g., alpha, delta, u, g, r, i, z, redshift).

  - Motivation: Modern telescopes produce large datasets, making manual classification impractical. Machine learning can efficiently identify and categorize     objects.

2. Data & Preprocessing

  - Dataset: Attributes like alpha, delta, u, g, r, i, z, redshift; target: class (GALAXY, QSO, STAR).

 - Preprocessing:

  - Mean imputation for missing values

  - Feature standardization (StandardScaler)

  - Polynomial feature expansion to capture non-linear interactions

3. Algorithms & Models

  - Logistic Regression – Baseline, interpretable.

  - K-Nearest Neighbors (KNN) – Non-parametric benchmark.

  - Random Forest – Ensemble method, good with non-linear data.

  - Gradient Boosting – Iterative boosting, emphasizes misclassified samples.

  - XGBoost – Optimized boosting; best performance in this project.

  - Decision Trees – Highly interpretable baseline.

4. Implementation Highlights

  - Language/Tools: Python, Pandas, Scikit-learn, XGBoost, Matplotlib/Seaborn.

  - Train/Test Split: 70% training, 30% testing.

  - Metrics: Accuracy & ROC-AUC used.

Observations:

  - XGBoost yielded the highest accuracy (~0.9756) and ROC-AUC (~0.9955).

  - Gradient Boosting & Random Forest also performed very well.

  - Logistic Regression scored ~0.9525 accuracy, ~0.9903 ROC-AUC.

5. Complexity Notes

  - Polynomial Features: Time & space grow exponentially with degree (O(n·d^k)), largest bottleneck.

Model Training:

  - Random Forest: O(T·m·log(n))

  - XGBoost/GBM: O(T·n·log(n))

  - Logistic Regression: O(n·d + d^2)

KNN: Expensive at prediction time (O(n_train·d)).

6. Top Features in XGBoost

  - redshift (0.8612)

  - g (0.0322)

  - z (0.0222)

  - u (0.0150)

  - g × z (0.0100)

7. Conclusion

  - XGBoost is recommended for large-scale stellar classification.

  - Improved efficiency and accuracy support astronomical research (e.g., galaxy evolution studies, star cataloging).
