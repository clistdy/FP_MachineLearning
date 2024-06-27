import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('Salesdata.csv')
df = df[~df['TransactionNo'].str.startswith('C')]
df['TotalSpent'] = df['Quantity'] * df['Price']
customer_spending = df.groupby('CustomerNo')['TotalSpent'].sum().reset_index()

low_spender_threshold = 3000
high_spender_threshold = 15000

def categorize_spending(total_spent):
    if total_spent < low_spender_threshold:
        return 'Low Spender'
    elif total_spent < high_spender_threshold:
        return 'Medium Spender'
    else:
        return 'High Spender'

customer_spending['SpendingCategory'] = customer_spending['TotalSpent'].apply(categorize_spending)

# Split data into features and target
X = customer_spending[['TotalSpent']]
y = customer_spending['SpendingCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

dt_base = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_base.fit(X_train, y_train)
y_pred_base = dt_base.predict(X_test)

results['Base Decision Tree (No Scaling)'] = {
    'Accuracy': accuracy_score(y_test, y_pred_base),
    'Precision (macro)': precision_score(y_test, y_pred_base, average='macro'),
    'Recall (macro)': recall_score(y_test, y_pred_base, average='macro'),
    'F1 Score (macro)': f1_score(y_test, y_pred_base, average='macro'),
}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_scaled = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = dt_scaled.predict(X_test_scaled)

results['Decision Tree (With Scaling)'] = {
    'Accuracy': accuracy_score(y_test, y_pred_scaled),
    'Precision (macro)': precision_score(y_test, y_pred_scaled, average='macro'),
    'Recall (macro)': recall_score(y_test, y_pred_scaled, average='macro'),
    'F1 Score (macro)': f1_score(y_test, y_pred_scaled, average='macro'),
}

param_grid_tree = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
grid_tree = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42), param_grid_tree, cv=10, scoring='accuracy')
grid_tree.fit(X_train_scaled, y_train)

best_tree = grid_tree.best_estimator_
best_tree.fit(X_train_scaled, y_train)
y_pred_best_tree = best_tree.predict(X_test_scaled)

results['Best Decision Tree (Hyperparameter Tuned)'] = {
    'Accuracy': accuracy_score(y_test, y_pred_best_tree),
    'Precision (macro)': precision_score(y_test, y_pred_best_tree, average='macro'),
    'Recall (macro)': recall_score(y_test, y_pred_best_tree, average='macro'),
    'F1 Score (macro)': f1_score(y_test, y_pred_best_tree, average='macro'),
}

print(pd.DataFrame(results).T)

plt.figure(figsize=(12, 8))
plot_tree(best_tree, feature_names=['TotalSpentScaled'], class_names=best_tree.classes_, filled=True)
plt.title('Decision Tree for Customer Spending Categories')
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred_best_tree)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=best_tree.classes_, yticklabels=best_tree.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
