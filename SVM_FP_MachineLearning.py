import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Filter out transactions that start with 'C'
    df = df[~df['TransactionNo'].str.startswith('C')]
    df['TotalSpent'] = df['Quantity'] * df['Price']
    customer_spending = df.groupby('CustomerNo')['TotalSpent'].sum().reset_index()
    return customer_spending

# Function to scale data
def scale_data(df):
    scaler = StandardScaler()
    df['TotalSpentScaled'] = scaler.fit_transform(df[['TotalSpent']])
    return df, scaler

# Function to assign spending categories
def assign_spending_categories(data, low_threshold, high_threshold):
    data['SpendingCategory'] = pd.cut(data['TotalSpent'], bins=[-np.inf, low_threshold, high_threshold, np.inf],
                                      labels=['Low Spender', 'Moderate Spender', 'High Spender'], right=True)
    return data

# Function to format currency
def format_currency(x):
    return f'£{x:,.0f}'

# Function to summarize clusters
def summarize_clusters(data):
    summary = data.groupby('SpendingCategory', observed=False)['TotalSpent'].describe()
    summary = summary.round(0)
    summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']] = summary[
        ['mean', 'std', 'min', '25%', '50%', '75%', 'max']].map(format_currency)
    print(summary)

# Function to calculate additional metrics
def calculate_additional_metrics(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    return f1, recall, precision, accuracy, cm

# Function to visualize clusters with scatter plot
def visualize_clusters(data, labels):
    plt.scatter(data['TotalSpentScaled'], np.zeros_like(data['TotalSpentScaled']), c=labels, cmap='viridis', marker='o')
    plt.xlabel('TotalSpentScaled')
    plt.title('Cluster Visualization')
    plt.colorbar(label='Cluster')
    plt.show()

# Function to summarize spending categories
def summarize_spending_categories(data):
    summary = data.groupby('SpendingCategory', observed=False)['TotalSpent'].describe()
    var = summary is summary.round(0)
    summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']] = summary[
        ['mean', 'std', 'min', '25%', '50%', '75%', 'max']].map(format_currency)
    return summary

# Function to run test scenarios
def run_test_scenarios(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nData split: Train and Test sets created")
    print(f"\nX_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Scenario 1: Baseline Performance Comparison
    svc = SVC(kernel='linear', C=1, probability=True)
    svc.fit(X_train, y_train)
    y_test_pred = svc.predict(X_test)
    y_test_prob = svc.predict_proba(X_test)[:, 1]
    f1_test, recall_test, precision_test, accuracy_test, cm_test = calculate_additional_metrics(y_test, y_test_pred)
    print(f"Baseline Performance (Linear SVM):")
    print(f"F1 Score: {f1_test:.2f}")
    print(f"Recall: {recall_test:.2f}")
    print(f"Precision: {precision_test:.2f}")
    print(f"Accuracy: {accuracy_test:.2f}")
    print(f"Confusion Matrix:\n{cm_test}")

    # Scenario 2: Impact of Feature Scaling
    X_scaled, scaler = scale_data(X.copy())
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled[['TotalSpentScaled']], y, test_size=0.2,
                                                                      random_state=42)
    print(f"\nData scaled: X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")
    svc.fit(X_train_scaled, y_train)
    y_test_pred_scaled = svc.predict(X_test_scaled)
    y_test_prob_scaled = svc.predict_proba(X_test_scaled)[:, 1]
    f1_test_scaled, recall_test_scaled, precision_test_scaled, accuracy_test_scaled, cm_test_scaled = calculate_additional_metrics(
        y_test, y_test_pred_scaled)
    print(f"Impact of Feature Scaling (Linear SVM):")
    print(f"F1 Score: {f1_test_scaled:.2f}")
    print(f"Recall: {recall_test_scaled:.2f}")
    print(f"Precision: {precision_test_scaled:.2f}")
    print(f"Accuracy: {accuracy_test_scaled:.2f}")
    print(f"Confusion Matrix:\n{cm_test_scaled}")

    # Scenario 3: Hyperparameter Optimization
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVC(probability=True), param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train_scaled, y_train)
    best_svc = grid_search.best_estimator_
    y_test_pred_optimized = best_svc.predict(X_test_scaled)
    y_test_prob_optimized = best_svc.predict_proba(X_test_scaled)[:, 1]
    f1_test_optimized, recall_test_optimized, precision_test_optimized, accuracy_test_optimized, cm_test_optimized = calculate_additional_metrics(
        y_test, y_test_pred_optimized)
    print(f"\nHyperparameter Optimization (SVM):")
    print(f"F1 Score: {f1_test_optimized:.2f}")
    print(f"Recall: {recall_test_optimized:.2f}")
    print(f"Precision: {precision_test_optimized:.2f}")
    print(f"Accuracy: {accuracy_test_optimized:.2f}")
    print(f"Confusion Matrix:\n{cm_test_optimized}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Visualize clusters using the best labels on the test set
    visualize_clusters(pd.DataFrame(X_test_scaled, columns=['TotalSpentScaled']), y_test_pred_optimized)

def main():
    # Load and preprocess the data
    file_path = 'Sales Transaction v.4a.csv'
    customer_spending = load_and_preprocess_data(file_path)

    # Define spending thresholds
    low_spender_threshold = 3000
    high_spender_threshold = 15000

    # Assign spending categories
    customer_spending = assign_spending_categories(customer_spending, low_spender_threshold, high_spender_threshold)

    # Summarize spending categories
    spending_summary = summarize_spending_categories(customer_spending)
    print(spending_summary)

    # Create true labels for evaluation
    customer_spending['SpendingCategoryCode'] = customer_spending['SpendingCategory'].cat.codes
    X = customer_spending[['TotalSpent']]
    y = customer_spending['SpendingCategoryCode']

    # Run test scenarios
    run_test_scenarios(X, y)

if __name__ == "__main__":
    main()
