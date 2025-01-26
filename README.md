# Exploring Machine Learning Sampling Techniques and Model Evaluation

This project demonstrates the application of various **machine learning models** and **sampling techniques** to address **class imbalance** and assess model performance under different conditions. By analyzing the effectiveness of various combinations of sampling methods, sample sizes, and machine learning models, the goal is to identify the optimal approach for achieving the highest accuracy in predictive tasks.

---

## Key Features of the Project  

- **Machine Learning Models**: Logistic Regression, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Gradient Boosting.  
- **Sampling Methods**: Exploring different sampling fractions and **SMOTE** (Synthetic Minority Over-sampling Technique) to effectively address class imbalance.  
- **Model Evaluation**: Comparison of model performance across various sampling sizes and fractions, using accuracy as the key evaluation metric.

---

## Overview of Project Workflow  

### 1. **Data Acquisition and Loading**  
   - The dataset is loaded from a CSV file: `Creditcard_data.csv`.

### 2. **Data Cleaning and Preprocessing**  
   - Handling missing values and scaling features with `StandardScaler` to prepare the data for machine learning models.

### 3. **Handling Class Imbalance with SMOTE**  
   - **SMOTE** is used to create synthetic minority class samples, ensuring a balanced dataset for training models.

### 4. **Experimenting with Sampling Strategies**  
   - The dataset is divided into multiple sample sizes (e.g., from 15% to 55% of the balanced dataset).  
   - Various training fractions (e.g., 30%, 50%, 65%, 75%, and 90%) are explored to evaluate performance under different conditions.

### 5. **Model Training and Performance Evaluation**  
   - **Models Trained**:
     1. Logistic Regression  
     2. Random Forest  
     3. Support Vector Machine (SVM)  
     4. K-Nearest Neighbors (KNN)  
     5. Gradient Boosting  
   - Each model is trained on different samples and evaluated for accuracy using `accuracy_score`.

### 6. **Result Analysis and Comparison**  
   - Aggregated results to identify the best sampling method for each model, based on average accuracy across all configurations.

---

## Key Output Files  

1. **`Balanced_Creditcard_data.csv`**  
   - The dataset after applying **SMOTE** to balance the class distribution.

2. **`Sampling_Accuracy_Results.csv`**  
   - Detailed accuracy results for all combinations of sample size, sampling technique, and model.

3. **`Best_Sampling_Technique_per_model.csv`**  
   - The optimal sampling technique for each model, based on average accuracy.

---

## Step-by-Step Breakdown  

### 1. **Loading the Data**  
   The dataset is loaded from `Creditcard_data.csv`, ensuring the file is located in the correct directory.

### 2. **Preprocessing the Data**  
   - Missing values are removed for clean data.  
   - Feature scaling is done using `StandardScaler` to enhance model performance and convergence.

### 3. **Class Imbalance Handling**  
   - **SMOTE** is applied to generate synthetic samples for the minority class, ensuring the dataset is balanced. The balanced dataset is saved as `Balanced_Creditcard_data.csv`.

### 4. **Sampling Strategy**  
   - Different sample sizes (from 15% to 55%) are used, and training fractions range from 30% to 90% to evaluate performance across different sampling strategies.

### 5. **Training the Models**  
   - The five machine learning models are trained on the varied datasets, and accuracy is calculated for each configuration.  

### 6. **Result Compilation**  
   - The best sampling techniques for each model are identified and saved to `Best_Sampling_Technique_per_model.csv`.  
   - Detailed accuracy results for all combinations are saved to `Sampling_Accuracy_Results.csv`.

---

## System Requirements  

To run the project, ensure the following Python libraries are installed:

- `pandas`  
- `numpy`  
- `scikit-learn`  
- `imbalanced-learn`  

Use the following command to install dependencies:  

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

---

## Instructions for Use  

### 1. **Running the Script**  
   Execute the Python script to preprocess data, apply SMOTE, and train the models:  
   ```bash
   python main.py
   ```

### 2. **Reviewing the Results**  
   - View `Best_Sampling_Technique_per_model.csv` for the optimal sampling method for each model.  
   - Check `Sampling_Accuracy_Results.csv` for a comprehensive breakdown of accuracy results across different configurations.

---

## Example Results Summary  

### Best Sampling Techniques (Sample Results)  

| **Model**              | **Best Sampling Technique** | **Average Accuracy** |  
|------------------------|-----------------------------|----------------------|  
| Logistic Regression     | Sampling1                  | 70.12%               |  
| Random Forest           | Sampling3                  | 90.45%               |  
| Support Vector Machine  | Sampling1                  | 78.25%               |  
| K-Nearest Neighbors     | Sampling1                  | 81.25%               |  
| Gradient Boosting       | Sampling5                  | 68.72%               |  

---

## Important Notes  

- Ensure that `Creditcard_data.csv` is in the correct directory, or adjust the file path in the script as needed.  
- The `train_test_split` function is used to split the data into training and testing sets. The project involves repeated experiments across various sampling and model configurations to evaluate their robustness and performance.

---

## Acknowledgments  

This project uses:
- **scikit-learn** for implementing the machine learning models.  
- **imbalanced-learn** for the **SMOTE** technique used to balance the dataset.

Feel free to adapt, extend, and apply this project to your own use cases and datasets for a deeper understanding of how different sampling strategies and models perform under class imbalance.
```

This is the proper Markdown format for a `README.md` file. You can copy and paste it directly into a `.md` file.
