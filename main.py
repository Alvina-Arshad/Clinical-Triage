import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb  
from visualization import plot_gender_disease_distribution, plot_age_distribution, plot_blood_pressure_distribution, plot_cholesterol_distribution, plot_fatigue_distribution, plot_difficulty_breathing_distribution, plot_correlation_heatmap, plot_confusion_matrix, plot_roc_curve, plot_f1_f2_comparison
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# Data Cleaning and Preprocessing
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill missing numeric values with the median
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Fill missing categorical values with the mode
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

data.drop_duplicates(inplace=True)

# Encode 'Gender' column (Male -> 0, Female -> 1)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Encode 'Fatigue' and 'Difficulty Breathing' columns (Yes -> 1, No -> 0)
label_encoder = LabelEncoder()
data['Fatigue'] = label_encoder.fit_transform(data['Fatigue']) 
data['Difficulty Breathing'] = label_encoder.fit_transform(data['Difficulty Breathing']) 

# Encode 'Fever' and 'Cough' columns (Yes -> 1, No -> 0)
data['Fever'] = label_encoder.fit_transform(data['Fever'])  
data['Cough'] = label_encoder.fit_transform(data['Cough']) 

# Encode 'Blood Pressure' and 'Cholesterol Level' using Label Encoding
data['Blood Pressure'] = label_encoder.fit_transform(data['Blood Pressure']) 
data['Cholesterol Level'] = label_encoder.fit_transform(data['Cholesterol Level'])  

# Encode 'Disease' and 'Outcome Variable' columns using Label Encoding
data['Disease'] = label_encoder.fit_transform(data['Disease'])
data['Outcome Variable'] = label_encoder.fit_transform(data['Outcome Variable'])

# Step 1: Split the data into features (X) and target (y)
X = data.drop(columns=['Outcome Variable'])
y = data['Outcome Variable']

# Feature Scaling (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply SMOTE only after splitting the data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check the distribution of classes after resampling
print(f"Class distribution after SMOTE: {y_resampled.value_counts()}")

# Step 3: Binning Age into categories
bins = [0, 20, 40, 60, 80, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Elderly', 'Very Elderly']
data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels)

# One-hot encoding for Age Group
data = pd.get_dummies(data, columns=['Age Group'], drop_first=True)

# Check the first few rows to confirm the new feature
print(data.head())

# Step 4: Hyperparameter Tuning for Logistic Regression, Random Forest, and XGBoost

# Logistic Regression Hyperparameter Tuning
log_reg = LogisticRegression(random_state=42)
param_grid_log_reg = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200]
}

grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_log_reg.fit(X_resampled, y_resampled)

# Best parameters for Logistic Regression
print("Best Parameters for Logistic Regression: ", grid_search_log_reg.best_params_)
log_reg_best = grid_search_log_reg.best_estimator_

# Random Forest Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_resampled, y_resampled)

# Best parameters for Random Forest
print("Best Parameters for Random Forest: ", grid_search_rf.best_params_)
rf_best = grid_search_rf.best_estimator_

# XGBoost Hyperparameter Tuning
xgboost_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

grid_search_xgb = GridSearchCV(xgboost_model, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_resampled, y_resampled)

# Best parameters for XGBoost
print("Best Parameters for XGBoost: ", grid_search_xgb.best_params_)
xgb_best = grid_search_xgb.best_estimator_

# Step 5: Model Ensembling using Voting Classifier
from sklearn.ensemble import VotingClassifier

# Create a voting classifier with the best models
voting_clf = VotingClassifier(estimators=[('log_reg', log_reg_best), ('rf', rf_best), ('xgb', xgb_best)], voting='soft')

# Train the voting classifier
voting_clf.fit(X_resampled, y_resampled)
y_pred_voting = voting_clf.predict(X_test)

# Now, Predictions for Logistic Regression, Random Forest, and XGBoost can be done
y_pred_log_reg = log_reg_best.predict(X_test) 
y_pred_rf = rf_best.predict(X_test) 
y_pred_xgb = xgb_best.predict(X_test) 

# Evaluate Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

# Evaluate Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Evaluate XGBoost
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))


# Define the Neural Network (MLP)
nn = Sequential([
    Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the Neural Network model
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Precision', 'Recall'])

# Apply SMOTE on the training data
X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Early stopping to avoid overfitting
es = EarlyStopping(patience=5, restore_best_weights=True)

# Train the Neural Network model
nn.fit(X_sm, y_sm, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es], verbose=0)
# Get predicted probabilities for Neural Network
y_prob_nn = nn.predict(X_test).flatten()

# Convert probabilities to predicted labels (threshold 0.5)
y_pred_nn = (y_prob_nn >= 0.5).astype(int)

# Plot Precision-Recall for each model
precision_log_reg, recall_log_reg, _ = precision_recall_curve(y_test, log_reg_best.predict_proba(X_test)[:, 1])
plt.plot(recall_log_reg, precision_log_reg, label="Logistic Regression")

precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_best.predict_proba(X_test)[:, 1])
plt.plot(recall_rf, precision_rf, label="Random Forest")

precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_best.predict_proba(X_test)[:, 1])
plt.plot(recall_xgb, precision_xgb, label="XGBoost")

precision_nn, recall_nn, _ = precision_recall_curve(y_test, nn.predict(X_test).flatten())
plt.plot(recall_nn, precision_nn, label="Neural Network")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Step 7: Evaluate Model with AUC
from sklearn.metrics import roc_auc_score

# Logistic Regression - AUC
y_prob_log_reg = log_reg_best.predict_proba(X_test)[:, 1]
auc_log_reg = roc_auc_score(y_test, y_prob_log_reg)
print(f"Logistic Regression AUC: {auc_log_reg:.2f}")

# Random Forest - AUC
y_prob_rf = rf_best.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"Random Forest AUC: {auc_rf:.2f}")

# XGBoost - AUC
y_prob_xgb = xgb_best.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"XGBoost AUC: {auc_xgb:.2f}")

# Save the final model
joblib.dump(rf_best, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved model.pkl and scaler.pkl")

# Plot Confusion Matrix for all models
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
# plot_confusion_matrix(cm_log_reg, title="Logistic Regression Confusion Matrix")

cm_rf = confusion_matrix(y_test, y_pred_rf)
# plot_confusion_matrix(cm_rf, title="Random Forest Confusion Matrix")

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
# plot_confusion_matrix(cm_xgb, title="XGBoost Confusion Matrix")

# Plot ROC Curve for all models
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg_best.predict_proba(X_test)[:, 1])
roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)
# plot_roc_curve(fpr_log_reg, tpr_log_reg, roc_auc_log_reg, title="Logistic Regression ROC Curve")

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_best.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)
# plot_roc_curve(fpr_rf, tpr_rf, roc_auc_rf, title="Random Forest ROC Curve")

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_best.predict_proba(X_test)[:, 1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
# plot_roc_curve(fpr_xgb, tpr_xgb, roc_auc_xgb, title="XGBoost ROC Curve")

# Final Visualization: Plot F1 and F2 Scores
# Logistic Regression - F1 and F2 Scores
f1_log_reg = f1_score(y_test, y_pred_log_reg)
f2_log_reg = (5 * precision_score(y_test, y_pred_log_reg) * recall_score(y_test, y_pred_log_reg)) / (4 * precision_score(y_test, y_pred_log_reg) + recall_score(y_test, y_pred_log_reg))

# Random Forest - F1 and F2 Scores
f1_rf = f1_score(y_test, y_pred_rf)
f2_rf = (5 * precision_score(y_test, y_pred_rf) * recall_score(y_test, y_pred_rf)) / (4 * precision_score(y_test, y_pred_rf) + recall_score(y_test, y_pred_rf))

# XGBoost - F1 and F2 Scores
f1_xgb = f1_score(y_test, y_pred_xgb)
f2_xgb = (5 * precision_score(y_test, y_pred_xgb) * recall_score(y_test, y_pred_xgb)) / (4 * precision_score(y_test, y_pred_xgb) + recall_score(y_test, y_pred_xgb))

# ──────────────────────────────────────────────────────────────────────────────
# Section: Four Accuracy Tests + Simple Neural Net
# ──────────────────────────────────────────────────────────────────────────────


def compute_f2(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return (5 * p * r) / (4 * p + r) if (p + r) > 0 else 0

print("\n\n=== FOUR ACCURACY TESTS ===")

# 1) Train on 20%, test on 80%  &  2) Train on 80%, test on 20%
for train_frac in [0.2, 0.8]:
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, train_size=train_frac, random_state=42)
    X_r, y_r = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    # Copy rf_best params and remove random_state to avoid duplication
    rf_params = rf_best.get_params().copy()
    rf_params.pop('random_state', None)

    rf_temp = RandomForestClassifier(**rf_params, random_state=42)
    rf_temp.fit(X_r, y_r)

    y_hat = rf_temp.predict(X_te)
    f1 = f1_score(y_te, y_hat)
    f2 = compute_f2(y_te, y_hat)
    print(f" Train {int(train_frac*100)}% → Test {100-int(train_frac*100)}% : F1={f1:.3f}, F2={f2:.3f}")

# 3) Repeated random 80/20 splits
repeats = 10
f1s, f2s = [], []
for seed in range(repeats):
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=seed)
    X_r, y_r = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)

    rf_params = rf_best.get_params().copy()
    rf_params.pop('random_state', None)

    rf_temp = RandomForestClassifier(**rf_params, random_state=seed)
    rf_temp.fit(X_r, y_r)

    y_hat = rf_temp.predict(X_te)
    f1s.append(f1_score(y_te, y_hat))
    f2s.append(compute_f2(y_te, y_hat))

print(f"\n Random-split (80/20) over {repeats} runs:")
print("  F1 scores:", np.round(f1s, 3))
print("  Mean ± SD:", np.mean(f1s), "±", np.std(f1s))
print("  F2 scores:", np.round(f2s, 3))
print("  Mean ± SD:", np.mean(f2s), "±", np.std(f2s))

# 4) Simple feed-forward neural network
print("\n=== SIMPLE NEURAL NETWORK (TensorFlow Keras) ===")
nn = Sequential([
    Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['Precision','Recall']
)

X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
es = EarlyStopping(patience=5, restore_best_weights=True)
nn.fit(X_sm, y_sm, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es], verbose=0)

loss, prec_nn, rec_nn = nn.evaluate(X_test, y_test, verbose=0)
f1_nn = 2 * (prec_nn * rec_nn) / (prec_nn + rec_nn) if (prec_nn + rec_nn) > 0 else 0
f2_nn = compute_f2(y_test, (nn.predict(X_test)[:,0] >= 0.5).astype(int))
y_prob_nn = nn.predict(X_test).flatten()
y_pred_nn = (y_prob_nn >= 0.5).astype(int)
cm_nn = confusion_matrix(y_test, y_pred_nn)
# plot_confusion_matrix(cm_nn, title="Neural Net Confusion Matrix")
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)
# plot_roc_curve(fpr_nn, tpr_nn, roc_auc_nn, title="Neural Net ROC Curve")

print(f" Neural Net — Precision: {prec_nn:.3f}, Recall: {rec_nn:.3f}, F1: {f1_nn:.3f}, F2: {f2_nn:.3f}")

# Plot F1/F2 for all four models
models = ['LogReg','RF','XGB','NN']
f1s_all = [f1_log_reg, f1_rf, f1_xgb, f1_nn]
f2s_all = [f2_log_reg, f2_rf, f2_xgb, f2_nn]

fig, ax = plt.subplots(figsize=(8,6))
x = range(len(models))
ax.bar(x,                f1s_all, width=0.4, label='F1', align='center', alpha=0.7)
ax.bar([i+0.4 for i in x], f2s_all, width=0.4, label='F2', align='center', alpha=0.7)
ax.set_xticks([i+0.2 for i in x])
ax.set_xticklabels(models)
ax.set_ylabel('Score')
ax.set_title('F1 vs. F2 Comparison (incl. Neural Net)')
ax.legend()
plt.tight_layout()
plt.show()

######################
# Define train-test splits to evaluate
split_ratios = [0.8, 0.7, 0.6, 0.5]
f1_scores = []
f2_scores = []

# Evaluate the models on different splits
for train_size in split_ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=train_size, random_state=42)
    
    # Resample using SMOTE
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Train each model and evaluate performance
    models = [log_reg_best, rf_best, xgb_best, nn]
    f1_split = []
    f2_split = []
    for model in models:
        # Get predicted probabilities for Neural Network
        if model == nn:
            y_prob = model.predict(X_test).flatten()
            y_pred = (y_prob >= 0.5).astype(int) 
        else:
            y_pred = model.predict(X_test) 

        # F1 Score
        f1 = f1_score(y_test, y_pred)
        
        # F2 Score
        f2 = (5 * precision_score(y_test, y_pred) * recall_score(y_test, y_pred)) / (4 * precision_score(y_test, y_pred) + recall_score(y_test, y_pred))
        
        # Append the results
        f1_split.append(f1)
        f2_split.append(f2)
    
    f1_scores.append(f1_split)
    f2_scores.append(f2_split)

# Convert to numpy arrays for plotting
f1_scores = np.array(f1_scores)
f2_scores = np.array(f2_scores)
models_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network']

# Plotting the F1 and F2 scores for each model
fig, ax = plt.subplots(figsize=(10,6))
for i, model_name in enumerate(models_names):
    ax.plot(split_ratios, f1_scores[:, i], marker='o', label=f'{model_name} - F1')
    ax.plot(split_ratios, f2_scores[:, i], marker='x', label=f'{model_name} - F2')

ax.set_xlabel('Train-Test Split Ratio')
ax.set_ylabel('Score')
ax.set_title('Impact of Data Splitting on Model Performance')
ax.legend()
plt.show()

# Plot F1 and F2 Scores for all models
# Visualizations (Call functions from visualization.py)

plot_gender_disease_distribution(data)
plot_age_distribution(data)
plot_blood_pressure_distribution(data)
plot_cholesterol_distribution(data)
plot_fatigue_distribution(data)
plot_difficulty_breathing_distribution(data)
plot_correlation_heatmap(data)
# plot_f1_f2_comparison(f1_log_reg, f2_log_reg, f1_rf, f2_rf, f1_xgb, f2_xgb)


# Add predictions to the original data
data['Logistic_Regression_Pred'] = log_reg_best.predict(X_scaled)
data['Random_Forest_Pred'] = rf_best.predict(X_scaled)
data['XGBoost_Pred'] = xgb_best.predict(X_scaled)
data['Neural_Network_Pred'] = (nn.predict(X_scaled).flatten() >= 0.5).astype(int)

# Save the updated data with predictions to a new CSV file
data.to_csv('Disease_symptom_and_patient_profile_with_predictions.csv', index=False)

# Compare predictions with the actual outcome variable
data['Logistic_Regression_Comparison'] = data['Logistic_Regression_Pred'] == data['Outcome Variable']
data['Random_Forest_Comparison'] = data['Random_Forest_Pred'] == data['Outcome Variable']
data['XGBoost_Comparison'] = data['XGBoost_Pred'] == data['Outcome Variable']
data['Neural_Network_Comparison'] = data['Neural_Network_Pred'] == data['Outcome Variable']

# Save the updated data with comparison results to a new CSV file
data.to_csv('Disease_symptom_and_patient_profile_with_comparison.csv', index=False)


