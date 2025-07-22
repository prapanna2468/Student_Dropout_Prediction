# Student_Dropout_Prediction
This project focuses on preprocessing and analyzing a synthetic educational dataset representing student data from the America Nepal Education Foundation (ANEF) for the Bagmati region in 2024. The dataset includes features such as student demographics, family income, academic performance, and dropout status. The goal is to clean, preprocess, and prepare the data for machine learning tasks, such as predicting student dropout risk, to derive actionable insights for educational interventions.

Theoretical Description
This repository contains a Jupyter Notebook (Individual_Assignment.ipynb) that implements a comprehensive data preprocessing pipeline for a synthetic educational dataset. Below is a theoretical overview of the key steps and methodologies employed:

1. Libraries Used
The project leverages several Python libraries for data manipulation, preprocessing, and visualization:

Pandas: For data manipulation and handling in DataFrame structures.
NumPy: For numerical operations and random number generation.
Scikit-learn: For data preprocessing techniques like normalization.
Faker: For generating synthetic data to simulate realistic student records.
Random: For controlling randomness and ensuring reproducibility.
Matplotlib/Seaborn (inferred): For visualizing results, such as the confusion matrix for model evaluation.
2. Data Generation
A synthetic dataset is created to simulate real-world educational data with 1,500 records. The dataset includes the following features:

Student_ID: Unique identifier for each student (e.g., STU####).
Age: Randomly generated ages between 11 and 18 years.
Gender: Binary categorical variable (Male/Female) with a slight bias toward males (52% probability).
Family_Income: Normally distributed income values, clipped between $5,000 and $50,000.
Parental_Education: Categorical variable representing parental education levels (None, Primary, Secondary, Higher).
Distance_to_School: Uniformly distributed distance values between 0.5 and 15 kilometers.
Attendance_Rate: Uniformly distributed attendance percentages between 20% and 100%.
Academic_Score: Uniformly distributed scores between 0 and 100.
Extracurricular_Participation: Binary categorical variable (Yes/No) indicating participation in extracurricular activities.
Dropout_Status: Binary target variable (Yes/No) indicating whether a student dropped out, with an 85% probability of "No."
School_ID: Identifier for schools (SCH01 to SCH50).
The dataset is generated with controlled randomness using a fixed seed (np.random.seed(42)) to ensure reproducibility.

3. Data Preprocessing
The preprocessing pipeline includes several steps to clean and prepare the dataset for analysis or modeling:

a. Introducing Missing Values
To simulate real-world data imperfections, approximately 2% of the values in Family_Income, Attendance_Rate, and Parental_Education are randomly set to missing (NaN). This mimics common issues in educational datasets, such as incomplete reporting.

b. Handling Missing Values
Missing values are imputed using the following strategies:

Family_Income: Replaced with the median income to account for potential skewness in income distribution.
Attendance_Rate: Imputed with the mean attendance rate for each School_ID to preserve school-specific patterns.
Parental_Education: Filled with the mode (most frequent category) to maintain the categorical distribution.
c. Standardizing Categorical Variables
Categorical variables are standardized to ensure consistency:

Gender: Any abbreviations (e.g., 'M', 'F') are mapped to 'Male' and 'Female'.
Parental_Education: Abbreviations like 'Sec' or 'Prim' are converted to 'Secondary' and 'Primary', respectively.
d. Encoding Categorical Variables
To make the data suitable for machine learning, categorical variables are encoded:

Gender: Mapped to binary values (Male: 1, Female: 0).
Extracurricular_Participation: Encoded as binary (Yes: 1, No: 0).
Parental_Education: Ordinal encoding based on education level (None: 0, Primary: 1, Secondary: 2, Higher: 3).
Dropout_Status: Binary encoding (Yes: 1, No: 0).
e. Normalization
Numerical features (Family_Income, Distance_to_School, Attendance_Rate, Academic_Score) are normalized using MinMaxScaler from Scikit-learn, scaling values to the range [0, 1]. This ensures that features with different scales (e.g., income vs. distance) contribute equally to machine learning models.

f. Dropping Unnecessary Columns
The Student_ID column is dropped as it is a unique identifier and does not contribute to predictive modeling.

g. Saving Preprocessed Data
The raw dataset is saved as bagmati_education_insights_2024.csv, and the preprocessed dataset is saved as preprocessed_bagmati_education.csv for further analysis or modeling.

4. Model Evaluation (Random Forest)
Although the provided code snippet includes references to a Random Forest model (e.g., confusion matrix visualization and metrics like accuracy, F1-score, and ROC-AUC), the model training code is not shown. The theoretical steps for this section include:

Model Training: A Random Forest classifier is likely trained on the preprocessed dataset to predict Dropout_Status.
Evaluation Metrics:
Accuracy: Measures the proportion of correct predictions.
F1-Score: Balances precision and recall, useful for imbalanced datasets.
ROC-AUC: Evaluates the model's ability to distinguish between dropout and non-dropout cases.
Confusion Matrix: Visualizes true positives, true negatives, false positives, and false negatives.
Cross-Validation: Computes mean and standard deviation of accuracy across multiple folds to assess model robustness.
Visualization: A confusion matrix is plotted using Seaborn's heatmap, saved as confusion_matrix.png.
Predictions: Model predictions are saved to rf_predictions_new.csv for verification.
5. Key Outputs
Raw Dataset: bagmati_education_insights_2024.csv
Preprocessed Dataset: preprocessed_bagmati_education.csv
Confusion Matrix Visualization: confusion_matrix.png
Model Predictions: rf_predictions_new.csv
Printed Metrics: Accuracy, F1-Score, ROC-AUC, confusion matrix, and cross-validation results.
Project Structure
Individual_Assignment.ipynb: Main Jupyter Notebook containing the data generation, preprocessing, and model evaluation code.
bagmati_education_insights_2024.csv: Synthetic raw dataset.
preprocessed_bagmati_education.csv: Cleaned and preprocessed dataset.
confusion_matrix.png: Visualization of the Random Forest model's confusion matrix.
rf_predictions_new.csv: Predicted dropout statuses from the Random Forest model.
Usage
To replicate the analysis:

Clone this repository.
Install the required Python libraries (pandas, numpy, scikit-learn, faker, matplotlib, seaborn).
Run the Individual_Assignment.ipynb notebook in a Jupyter environment.
Ensure the working directory has write permissions to save output files.
Future Improvements
Additional Models: Incorporate other machine learning models (e.g., Logistic Regression, XGBoost) for comparison.
Feature Engineering: Create new features, such as interaction terms or school-level aggregates, to improve model performance.
Handling Imbalanced Data: Apply techniques like SMOTE or class weighting to address the imbalance in Dropout_Status.
Advanced Visualizations: Include more exploratory data analysis (EDA) plots to uncover patterns in the data.
License
This project is licensed under the MIT License. See the LICENSE file for details.

This README provides a theoretical overview of the data preprocessing and analysis pipeline without reproducing code or results, as per the requirements. For detailed implementation, refer to the Individual_Assignment.ipynb notebook in the repository.
