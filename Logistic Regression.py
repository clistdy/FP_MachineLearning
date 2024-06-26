import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from google.colab import drive

#mount to dataset
drive.mount('/gdrive')
csv_path = '/gdrive/My Drive/ML/sales transactions.csv'
df = pd.read_csv(csv_path, encoding='latin1')
df.head()

 # Filter out transactions that start with 'C'
df = df[~df['TransactionNo'].str.startswith('C')]

# Calculate total spent per transaction
df['TotalSpent'] = df['Quantity'] * df['Price']

def categorize_spending(total_spent):
    low_spender_threshold = 3000
    high_spender_threshold = 15000
    
    if total_spent <= low_spender_threshold:
        return 'low'
    elif low_spender_threshold < total_spent <= high_spender_threshold:
        return 'medium'
    else:
        return 'high'

df['SpendingCategory'] = df['TotalSpent'].apply(categorize_spending)

# Show the first 10 rows
df.head(10)

# Features and labels
features = df[['Quantity', 'Price']]
labels = df['SpendingCategory']

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.3, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate accuracy
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (weighted): {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(classification_report(y_test, y_pred))

# Predict probabilities
y_prob = model.predict_proba(X_test)

# Add predicted categories to X_test
X_test = X_test.copy()
X_test['PredictedCategory'] = label_encoder.inverse_transform(y_pred)

# Get indices for each class
low_index = label_encoder.transform(['low'])[0]
medium_index = label_encoder.transform(['medium'])[0]
high_index = label_encoder.transform(['high'])[0]

# Create summary statistics for customer spending
customer_spending = df.groupby('CustomerNo')['TotalSpent'].sum().reset_index()
customer_spending['SpendingCategory'] = customer_spending['TotalSpent'].apply(categorize_spending)
summary = customer_spending.groupby('SpendingCategory')['TotalSpent'].describe()
summary = summary.round(0)

# Define a function to format currency
def format_currency(x):
    return f'€{x:,.0f}'

summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']] = summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].applymap(format_currency)
print(summary)

# Visualization
plt.figure(figsize=(14, 7))
sns.scatterplot(x='Quantity', y='Price', hue='PredictedCategory', data=X_test, palette='viridis')
plt.title('Predicted Spending Categories')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.legend(title='Spending Category')
plt.show()

# Convert probabilities to percentages
X_test['LowProb (%)'] = y_prob[:, low_index] * 100
X_test['MediumProb (%)'] = y_prob[:, medium_index] * 100
X_test['HighProb (%)'] = y_prob[:, high_index] * 100

print(X_test[['LowProb (%)', 'MediumProb (%)', 'HighProb (%)']])

# Calculate the mean of probabilities in percentages
mean_low_prob = X_test['LowProb (%)'].mean()
mean_medium_prob = X_test['MediumProb (%)'].mean()
mean_high_prob = X_test['HighProb (%)'].mean()

# Print the results
print(f"The Probability of Each Categories :")
print(f"LowProb (%): {mean_low_prob:.2f}")
print(f"MediumProb (%): {mean_medium_prob:.2f}")
print(f"HighProb (%): {mean_high_prob:.2f}")

#Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# Plot confusion matrix
class_names = ['low', 'medium', 'high']
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#Testing

#Baseline 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier

# Assuming features and labels are defined earlier in your code
# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.3, random_state=42)

# Baseline model (most frequent class)
baseline_model = DummyClassifier(strategy='most_frequent', random_state=42)
baseline_model.fit(X_train, y_train)
y_baseline_pred = baseline_model.predict(X_test)

# Evaluate Baseline model with macro-averaging
baseline_f1_macro = f1_score(y_test, y_baseline_pred, average='macro', zero_division=1)
baseline_accuracy_macro = accuracy_score(y_test, y_baseline_pred)
baseline_precision_macro = precision_score(y_test, y_baseline_pred, average='macro', zero_division=1)
baseline_recall_macro = recall_score(y_test, y_baseline_pred, average='macro', zero_division=1)

print("Baseline Model Performance (Macro-average):")
print(f"F1 Score (macro): {baseline_f1_macro:.4f}")
print(f"Accuracy: {baseline_accuracy_macro:.4f}")
print(f"Precision (macro): {baseline_precision_macro:.4f}")
print(f"Recall (macro): {baseline_recall_macro:.4f}")
print(classification_report(y_test, y_baseline_pred, target_names=label_encoder.classes_, zero_division=1))

#Feature Scaling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# Assuming features and labels are defined earlier in your code
# Feature scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.3, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate metrics using macro-averaging
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Feature Scaling Performance (Macro-average):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")

# Classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#Hyperparameter Optimization testing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# Assuming features and labels are defined earlier in your code
# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter optimization using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 500, 1000]
}

# Create logistic regression model
model = LogisticRegression(multi_class='multinomial', random_state=42)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test_scaled)

# Evaluate metrics using macro-averaging
f1_macro = f1_score(y_test, y_pred, average='macro')
accuracy_macro = accuracy_score(y_test, y_pred)
recall_macro = recall_score(y_test, y_pred, average='macro')
precision_macro = precision_score(y_test, y_pred, average='macro')

print(f"Best Parameters: {grid_search.best_params_}")
print(f"F1 Score (macro): {f1_macro:.4f}")
print(f"Accuracy: {accuracy_macro:.4f}")
print(f"Recall: {recall_macro:.4f}")
print(f"Precision: {precision_macro:.4f}")
print(classification_report(y_test, y_pred))

#Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Calculate total spent per transaction
df['TotalSpent'] = df['Quantity'] * df['Price']

# Define spending categories based on thresholds
def categorize_spending(total_spent):
    if total_spent <= 3000:
        return 'low'
    elif 3001 <= total_spent <= 17499:
        return 'medium'
    else:
        return 'high'

df['SpendingCategory'] = df['TotalSpent'].apply(categorize_spending)

# Features and target
features = df[['Quantity', 'Price']]
target = df['TotalSpent']
labels = df['SpendingCategory']

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(features, target, labels_encoded, test_size=0.3, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Categorize predictions
predicted_categories = [categorize_spending(y) for y in y_pred]
predicted_categories_encoded = label_encoder.transform(predicted_categories)

# Evaluate accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")


# Calculate F1 score
f1 = f1_score(y_test_labels, predicted_categories_encoded, average='weighted')
print(f"F1 Score (weighted): {f1:.4f}")
print(classification_report(y_test_labels, predicted_categories_encoded, target_names=label_encoder.classes_))

# Visualization
plt.figure(figsize=(14, 7))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)  # Plot actual vs predicted
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Plot ideal line
plt.title('Actual vs Predicted Total Spent')
plt.xlabel('Actual Total Spent')
plt.ylabel('Predicted Total Spent')
plt.show()
