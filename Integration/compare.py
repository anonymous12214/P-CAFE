from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def decision_tree_DiabeticRetinopathyDebrecen():
    # Splitting data into training and testing sets
    # fetch dataset
    diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
    X = diabetic_retinopathy_debrecen.data.features
    y = diabetic_retinopathy_debrecen.data.targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=15)
    clf.fit(X_train, y_train)
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy DT: {accuracy}")
    print(f"Tree Depth: {clf.get_depth()}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

def XGBOOST_DiabeticRetinopathyDebrecen():
    diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
    X = diabetic_retinopathy_debrecen.data.features
    #remove column number 10
    X = X.drop(X.columns[10], axis=1)
    y = diabetic_retinopathy_debrecen.data.targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #run XGBOOST
    clf = XGBClassifier(max_depth=9)
    clf.fit(X_train, y_train)
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy XGBOOST: {accuracy}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)


def decision_tree_mimic_time_series():
    X_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\train_X.csv')
    Y_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\train_Y.csv')
    X_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\val_X.csv')
    Y_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\val_Y.csv')
    X_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\test_X.csv')
    Y_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\test_Y.csv')
    X = pd.concat([X_train, X_val, X_test])
    Y = pd.concat([Y_train, Y_val, Y_test])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # Evaluate the model
    #calculate AUC-ROC score
    y_pred = clf.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC DT: {auc_roc}")
    auc_pr = average_precision_score(y_test, y_pred)
    print(f"AUC-PR: {auc_pr}")
    return auc_roc, auc_pr



def XGBOOST_mimic_time_series():
    X_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\train_X.csv')
    Y_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\train_Y.csv')
    X_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\val_X.csv')
    Y_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\val_Y.csv')
    X_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\test_X.csv')
    Y_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\mimic3-benchmarks\test_Y.csv')
    X = pd.concat([X_train, X_val, X_test])
    Y = pd.concat([Y_train, Y_val, Y_test])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    # Evaluate the model
    #calculate AUC-ROC score
    y_pred = clf.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC DT: {auc_roc}")
    #print AUC-PR score
    auc_pr = average_precision_score(y_test, y_pred)
    print(f"AUC-PR: {auc_pr}")
    # Get feature importance
    feature_importance = clf.feature_importances_
    num_features_used = np.count_nonzero(feature_importance > 0)

    print(f"Number of features used in classification: {num_features_used}")

    return auc_roc, auc_pr


def XGBOOST_mimic_no_text():
    path=r'C:\Users\kashann\PycharmProjects\mimic-code-extract\mimic-iii\notebooks\ipynb_example\data_numeric.json'
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    X = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    # Evaluate the model
    #calculate AUC-ROC score
    y_pred = clf.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc_roc}")

    auc_pr = average_precision_score(y_test, y_pred)
    print(f"AUC-PR: {auc_pr}")

def DT_mimic_no_text():
    path=r'C:\Users\kashann\PycharmProjects\mimic-code-extract\mimic-iii\notebooks\ipynb_example\data_numeric.json'
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    X = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # Evaluate the model
    #calculate AUC-ROC score
    y_pred = clf.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc_roc}")

    auc_pr = average_precision_score(y_test, y_pred)
    print(f"AUC-PR: {auc_pr}")



def IG():
    # Load data
    X_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\train_X.csv')
    Y_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\train_y.csv')
    X_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\val_X.csv')
    Y_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\val_y.csv')
    X_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\test_X.csv')
    Y_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\test_y.csv')

    # Combine datasets
    X = pd.concat([X_train, X_val, X_test])
    Y = pd.concat([Y_train, Y_val, Y_test])

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Calculate Mutual Information for feature selection
    mi_scores = mutual_info_classif(X_train, y_train.values.ravel(), random_state=42)

    # Select top features based on a threshold
    mi_threshold = np.percentile(mi_scores, 75)  # Top 25% features
    selected_features = X_train.columns[mi_scores >= mi_threshold]
    print(f"Selected features based on MI: {list(selected_features)}")

    # Filter datasets based on selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Train a classifier on selected features
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_selected, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test_selected)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc_roc}")
    auc_pr = average_precision_score(y_test, y_pred)
    print(f"AUC-PR: {auc_pr}")

    num_features_used = len(selected_features)
    print(f"Number of features used in classification: {num_features_used}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def RFE_time():
    # Load data
    X_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\train_X.csv')
    Y_train = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\train_y.csv')
    X_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\val_X.csv')
    Y_val = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\val_y.csv')
    X_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\test_X.csv')
    Y_test = pd.read_csv(r'C:\Users\kashann\PycharmProjects\PCAFE-MIMIC\input\data_time_series\test_y.csv')

    # Combine datasets
    X = pd.concat([X_train, X_val, X_test])
    Y = pd.concat([Y_train, Y_val, Y_test])

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Apply Recursive Feature Elimination (RFE)
    estimator = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=500)  # Select top 500
    rfe.fit(X_train, y_train.values.ravel())

    # Selected features
    selected_features = X_train.columns[rfe.support_]
    print(f"Selected features by RFE: {list(selected_features)}")

    # Filter datasets based on selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Train a classifier on selected features
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_selected, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test_selected)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc_roc}")
    auc_pr = average_precision_score(y_test, y_pred)
    print(f"AUC-PR: {auc_pr}")

    num_features_used = len(selected_features)
    print(f"Number of features used in classification: {num_features_used}")

    return auc_roc, auc_pr, num_features_used


def main():
    RFE_time()



if __name__ == '__main__':
    main()