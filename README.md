# Diabetes Data Classification

This project utilizes different machine learning algorithms to predict the diagnosis of diabetes based on a dataset of clinical features. The main goal is to compare the accuracy of three methods: Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM), applying preprocessing and evaluation techniques.

## Dataset

The dataset `diabetes.csv` contains information about patients, which are the variables used for training.

## Algorithms Used

1. **Decision Tree**: Uses a tree structure to divide the dataset into subsets based on more discriminative features.

2. **K-Nearest Neighbors (KNN)**: Classifies a new data point based on the most common class among its K nearest neighbors.

3. **Support Vector Machines (SVM)**: Finds the hyperplane that best separates the classes of data.

## Code Structure

1. **Preprocessing**:
   - The dataset is loaded.
   - The data is split into training and testing sets (75% training, 25% testing) using `train_test_split`.

2. **Training and Evaluation**:
   - Each model (Decision Tree, KNN, and SVM) is trained with the training data.
   - Predictions are generated for the test set.
   - The confusion matrix is used to calculate the accuracy of each model.

## Model Accuracy

I could have used a function from the scikit-learn library, but I chose to create my own function for this purpose.

The accuracy of each model is calculated based on the confusion matrix, which compares the predictions with the actual results, and it is always the same since the 'random_state' argument was set to zero in the `train_test_split`, `DecisionTreeClassifier`, and `SVC` functions. The following results were obtained:

- **Decision Tree**: `71.875 %`
- **K-Nearest Neighbors**: `75.52083333333334 %`
- **Support Vector Machines**: `80.20833333333334 %`

## How to Run

1. Install the necessary libraries:
   ```bash
   pip install pandas numpy scikit-learn

2. Change file path:
data = pd.read_csv(">> SET FILE PATH HERE <</diabetes.csv", encoding="latin1")   

3. Run the code:
   ```bash
   python Data_Mining.py
   ```

## Conclusion

This project provides a clear comparison between three popular classification algorithms applied to a real-world diabetes diagnosis problem. It highlights the importance of choosing the appropriate model and evaluating performance based on metrics such as accuracy.
