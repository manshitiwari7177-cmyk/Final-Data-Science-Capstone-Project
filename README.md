# Final-Data-Science-Capstone-Project
This capstone project demonstrates end-to-end data science skills, including data cleaning, EDA, visualization, machine learning, model evaluation, and reporting using Python and industry-standard libraries.
Final Data Science Capstone Project
Overview
This capstone demonstrates an end-to-end data science and machine-learning workflow. It uses a synthetic student-performance dataset to explore data, clean missing values, visualize patterns, train models, compare performance, and document insights.
Objective
Showcase skills acquired during training/internship through a reproducible project including data preparation, exploratory analysis, visualization, machine learning, evaluation, and reporting.
Dataset
The script generates synthetic records containing:
- Study hours per day
- Attendance percentage
- Assignment average
- Previous exam score
- Extracurricular participation
- Final exam score (target)
This is simulated data, not real student data.
Requirements
Python 3.8+:
pip install pandas numpy matplotlib seaborn scikit-learn
Run
From the project directory:
python final_data_science_capstone.py
On Windows, if needed:
py final_data_science_capstone.py
Workflow
1. Generate a synthetic dataset.
2. Inspect dataset shape, data types, missing values, and summary statistics.
3. Remove duplicates and fill missing numerical values using medians.
4. Perform exploratory data analysis and create visualizations.
5. Split data into training and test sets.
6. Train Linear Regression, Ridge Regression, and Random Forest models.
7. Compare models using MAE, RMSE, and R-squared.
8. Identify feature importance/coefficient magnitudes.
9. Save predictions, comparison metrics, charts, and a Markdown report.
Outputs
The script creates capstone_outputs containing:
- cleaned_dataset.csv
- model_comparison.csv
- test_predictions.csv
- capstone_report.md
- final_score_distribution.png
- study_hours_vs_score.png
- correlation_heatmap.png
- actual_vs_predicted.png
- feature_importance.png
Limitations and Responsible Use
The dataset is synthetic, so metrics are educational and do not prove performance on real data. In real use, obtain appropriate permissions, anonymize data, validate on representative samples, evaluate fairness, and avoid using predictions as the sole basis for high-impact decisions.
