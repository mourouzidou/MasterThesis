from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def preprocess_features(X):
    # Identify non-numeric columns and apply label encoding to categorical columns
    non_numeric_columns = X.select_dtypes(include=['object']).columns

    # Apply LabelEncoder to non-numeric columns
    for col in non_numeric_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[sorted_idx], align='center')
    plt.xticks(range(len(importance)), np.array(feature_names)[sorted_idx], rotation=90)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def impute_missing_values(X):
    # Impute missing values for numeric and non-numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')  # For numeric columns, fill with mean
    non_numeric_imputer = SimpleImputer(
        strategy='most_frequent')  # For non-numeric columns, fill with the most frequent value

    # Separate numeric and non-numeric columns
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_columns = X.select_dtypes(include=['object', 'category']).columns

    # Impute numeric columns
    X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])

    # Impute non-numeric columns
    if len(non_numeric_columns) > 0:
        X[non_numeric_columns] = non_numeric_imputer.fit_transform(X[non_numeric_columns])

    return X

def assign_most_frequent_phenotype(gene, predicted_df):
    """
    Assigns the most frequent phenotype to each unique genotype.
    """
    genotype_phenotype_pairs = predicted_df[[gene, f'{gene}_Predicted_Phenotype']]
    most_frequent_phenotype = genotype_phenotype_pairs.groupby(gene)[f'{gene}_Predicted_Phenotype'].apply(
        lambda x: Counter(x).most_common(1)[0][0]  # Get the most common phenotype for each genotype
    ).reset_index()

    # Save the result to a CSV file
    output_file = f'{gene}_unique_genotype_phenotype.csv'
    most_frequent_phenotype.to_csv(output_file, index=False)
    print(f"Saved unique genotype-phenotype assignments for {gene} to {output_file}")

    return most_frequent_phenotype

def run_classification_models(gene, merged_df):
    # Treat 'Indeterminate' as NaN (missing values)
    merged_df[f'{gene}_Phenotype'].replace(f'{gene} Indeterminate', np.nan, inplace=True)

    # Separate known and unknown diplotypes
    known_diplotypes_df = merged_df[merged_df[f'{gene}_Phenotype'].notna()]
    unknown_diplotypes_df = merged_df[merged_df[f'{gene}_Phenotype'].isna()]

    # If no unknown diplotypes, skip the prediction step
    if unknown_diplotypes_df.empty:
        print(f"All phenotypes for {gene} are known. Skipping prediction.")
        final_output = known_diplotypes_df[[gene, f'{gene}_Phenotype']].rename(
            columns={f'{gene}_Phenotype': f'{gene}_Predicted_Phenotype'})
        final_output.to_csv(f'{gene}_predicted_phenotypes.csv', index=False)
        return final_output

    # Prepare the feature matrix (X) and labels (y)
    X = known_diplotypes_df.drop(columns=[f'{gene}_Phenotype', gene, 'ParticipantID'] + [col for col in merged_df.columns if '_Phenotype' in col])
    y = known_diplotypes_df[f'{gene}_Phenotype']

    # Preprocess and impute missing values
    X = preprocess_features(X)
    X = impute_missing_values(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use RandomForest for classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict the unknown phenotypes
    X_unknown = unknown_diplotypes_df.drop(columns=[f'{gene}_Phenotype', gene, 'ParticipantID'] + [col for col in merged_df.columns if '_Phenotype' in col])
    X_unknown = preprocess_features(X_unknown)
    X_unknown = impute_missing_values(X_unknown)

    if X_unknown.shape[0] == 0:
        print(f"No unknown samples for prediction for {gene}. Skipping prediction.")
        return

    predicted_phenotypes = model.predict(X_unknown)
    unknown_diplotypes_df[f'{gene}_Predicted_Phenotype'] = predicted_phenotypes

    # Combine known and predicted phenotypes for final output
    final_output = pd.concat([
        known_diplotypes_df[[gene, f'{gene}_Phenotype']].rename(columns={f'{gene}_Phenotype': f'{gene}_Predicted_Phenotype'}),
        unknown_diplotypes_df[[gene, f'{gene}_Predicted_Phenotype']]
    ])

    # Save the final output
    final_output.to_csv(f'{gene}_predicted_phenotypes.csv', index=False)
    print(f"Predictions for {gene} saved to {gene}_predicted_phenotypes.csv")

    # Assign the most frequent phenotype for each genotype
    assign_most_frequent_phenotype(gene, final_output)

    return final_output

def run_pipeline_for_all_genes(merged_df, genes):
    for gene in genes:
        print(f"Running classification for {gene}")
        run_classification_models(gene, merged_df)

# Load the merged data and run the classification pipeline
merged_df = pd.read_csv('merged_training.csv')
genes = ['CYP2D6', 'CYP2C9', 'CYP2C19', 'TPMT', 'UGT1A1', 'CYP3A5', 'SLCO1B1', 'DPYD']

# run_pipeline_for_all_genes(merged_df, genes)
