# Day 2: Classification Models

## 📚 Learning Objectives

By the end of this day, you should understand:

- ✅ What is classification?
- ✅ Train multiple classification algorithms
- ✅ Evaluate model performance
- ✅ Compare different models
- ✅ Understand hyperparameters
- ✅ Make predictions on new data

## 🎯 Key Concepts

### 1. Classification Overview
Classification is supervised learning where we predict categorical labels.

**Types:**
- Binary classification (2 classes)
- Multi-class classification (>2 classes)

### 2. Common Algorithms

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| Logistic Regression | Linear separable data | Fast, interpretable | Limited to linear |
| Decision Tree | Non-linear patterns | Interpretable, no scaling | Overfitting prone |
| Random Forest | Complex patterns | Powerful, handles outliers | Black box |
| SVM | High-dimensional data | Effective in high dims | Slow on large datasets |
| Naive Bayes | Text/probability | Fast, probabilistic | Assumes independence |
| KNN | Small datasets | Simple, effective | Slow prediction |

### 3. Model Evaluation Metrics

- **Accuracy:** (TP+TN)/(TP+TN+FP+FN)
- **Precision:** TP/(TP+FP) - "How many predicted positive are actually positive?"
- **Recall:** TP/(TP+FN) - "How many actual positives did we find?"
- **F1-Score:** 2*(Precision*Recall)/(Precision+Recall) - Balanced metric
- **ROC-AUC:** Measure of classifier quality at various thresholds
- **Confusion Matrix:** TP, TN, FP, FN breakdown

### 4. Training Process

```
1. Load Data
2. Split Data (Train/Test)
3. Preprocess Data
4. Train Model(s)
5. Make Predictions
6. Evaluate Performance
7. Tune Hyperparameters
8. Compare Models
```

## 📊 Dataset

Using **Wine Quality Dataset** or **Titanic** (from Day 1).

## 🔧 Implementation

### Basic Classification Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Prepare data
X = df.drop('target', axis=1)
y = df['target']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```

### Comparing Multiple Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")
```

## 📝 Exercise

1. Load a classification dataset
2. Preprocess the data (handle missing values, scale features)
3. Train at least 3 different classification models
4. Evaluate each model using:
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
5. Visualize:
   - Confusion Matrix as heatmap
   - ROC curves for different models
   - Model comparison bar plot
6. Select best model and explain why

## 💡 Tips

1. **Always split data first:** Before any preprocessing
2. **Scale appropriately:** Most algorithms benefit from scaling
3. **Class imbalance:** Handle if one class is much larger
4. **Try multiple models:** No single best algorithm
5. **Look at errors:** Understand what model gets wrong

## 🔗 Resources

- [Scikit-learn Classification](https://scikit-learn.org/stable/modules/classification.html)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)

---

**Next:** Day 3 will cover advanced classification techniques!