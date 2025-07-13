import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# Function for Gender and Disease distribution
def plot_gender_disease_distribution(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gender distribution
    sns.countplot(x='Gender', data=data, ax=axes[0])
    axes[0].set_title('Gender Distribution')

    # Disease distribution
    sns.countplot(x='Disease', data=data, ax=axes[1])
    axes[1].set_title('Disease Distribution')

    plt.tight_layout()
    plt.show()

# Function for Age Distribution
def plot_age_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Age'], kde=True, bins=20)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

# Function for Blood Pressure Distribution
def plot_blood_pressure_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Blood Pressure', data=data)
    plt.title('Blood Pressure Distribution')
    plt.xlabel('Blood Pressure')
    plt.ylabel('Count')
    plt.show()

# Function for Cholesterol Level Distribution
def plot_cholesterol_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cholesterol Level', data=data)
    plt.title('Cholesterol Level Distribution')
    plt.xlabel('Cholesterol Level')
    plt.ylabel('Count')
    plt.show()

# Function for Fatigue Distribution
def plot_fatigue_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Fatigue', data=data)
    plt.title('Fatigue Distribution')
    plt.xlabel('Fatigue')
    plt.ylabel('Count')
    plt.show()

# Function for Difficulty Breathing Distribution
def plot_difficulty_breathing_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Difficulty Breathing', data=data)
    plt.title('Difficulty Breathing Distribution')
    plt.xlabel('Difficulty Breathing')
    plt.ylabel('Count')
    plt.show()

# Function for Correlation Heatmap
def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


# Function for plotting Confusion Matrix
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function for plotting ROC Curve
def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve"):
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

# Plot F1 and F2 Scores for all models
def plot_f1_f2_comparison(f1_log_reg, f2_log_reg, f1_rf, f2_rf, f1_xgb, f2_xgb):
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    f1_scores = [f1_log_reg, f1_rf, f1_xgb]
    f2_scores = [f2_log_reg, f2_rf, f2_xgb]

    x = range(len(models))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot F1 and F2 Scores
    ax.bar(x, f1_scores, width=0.4, label='F1 Score', align='center', alpha=0.7)
    ax.bar(x, f2_scores, width=0.4, label='F2 Score', align='edge', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Scores')
    ax.set_title('F1 and F2 Scores Comparison')
    ax.legend()

    plt.tight_layout()
    plt.show()
