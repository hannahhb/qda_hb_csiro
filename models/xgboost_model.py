import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD

# Step 1: Load the dataset
df1 = pd.read_excel('data/1.0 SLR CFIR Thematic Analysis - FINAL.xlsx')
df2 = pd.read_excel('data/2.0 SLR2 CFIR Thematic Analysis - FINAL.xlsx')
df = pd.concat([df1, df2], ignore_index=True)

# Assuming the dataset has two columns: 'Comment' (text) and 'Domain' (label)
comments = df['Comments']
domains = df['Domain']

# Step 2: Preprocessing the text data
# Convert text to TF-IDF features with n-grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
X = tfidf_vectorizer.fit_transform(comments)

# Step 3: Encoding the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(domains)

# Step 4: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 5: Dimensionality reduction (Optional, can improve performance)
svd = TruncatedSVD(n_components=500)
X_reduced = svd.fit_transform(X_resampled)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_resampled, test_size=0.2, random_state=42)

# Step 7: Train the XGBoost model with hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [500],
    'min_child_weight': [1, 5, 10]
}

xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(label_encoder.classes_), eval_metric='mlogloss')

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Step 8: Make predictions
y_pred = np.argmax(best_model.predict_proba(X_test), axis=1)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Optional: Save the model and vectorizer for future use
import joblib
joblib.dump(best_model, 'xgboost_cfir_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
