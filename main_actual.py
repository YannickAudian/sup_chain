import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

# Chargement des données
df = pd.read_csv('df_09_avec_category_nps.csv')

# Prétraitement des données
df_clean = df[['category_nps', 'nb_tokens']].dropna()

def reassign_neutral(row):
    if row['category_nps'] == 'Neutre':
        if row['nb_tokens'] < 3:
            return 'Détracteur'
        else:
            return 'Promoteur'
    else:
        return row['category_nps']

df_clean['category_nps'] = df_clean.apply(reassign_neutral, axis=1)

X = df_clean[['nb_tokens']].values
y = df_clean['category_nps'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),  # SVC with probability=True for ROC
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
}

results_file = open("resultats.txt", "w")
results_file.write("Résultats des différents modèles\n")
results_file.write("="*50 + "\n")

for model_name, model in models.items():
    pipeline = Pipeline([('transformer', BaseEstimator()), ('classifier', model)])
    param_grid = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    y_pred = grid_search.best_estimator_.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results_file.write(f"Modèle : {model_name}\n")
    results_file.write(f"Précision : {accuracy:.4f}\n")
    results_file.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Écriture des valeurs réelles vs prédites dans le fichier texte
    results_file.write("\nValeurs réelles vs prédites :\n")
    for real, pred in zip(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred)):
        results_file.write(f"Réel : {real}, Prédit : {pred}\n")
    results_file.write("\n" + "="*50 + "\n")
    
    # Matrice de confusion
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Matrice de confusion : {model_name}")
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Valeurs réelles")
    plt.savefig(f"matrice_confusion_{model_name}.png")
    plt.close()

    # Export des valeurs réelles et prédites
    comparison = pd.DataFrame({
        'Valeurs réelles': label_encoder.inverse_transform(y_test),
        'Valeurs prédites': label_encoder.inverse_transform(y_pred)
    })
    comparison.to_csv(f"reel_vs_predit_{model_name}.csv", index=False)

# Réseau de neurones
mlp_pipeline = Pipeline([
    ('transformer', BaseEstimator()),
    ('classifier', MLPClassifier(max_iter=500, random_state=42))
])

mlp_pipeline.fit(X_train, y_train)
y_pred_mlp = mlp_pipeline.predict(X_test)
proba_mlp = mlp_pipeline.predict_proba(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

results_file.write(f"Réseau de neurones (MLP)\n")
results_file.write(f"Précision : {accuracy_mlp:.4f}\n")
results_file.write(classification_report(y_test, y_pred_mlp, target_names=label_encoder.classes_))

# Écriture des valeurs réelles vs prédites pour MLP dans le fichier texte
results_file.write("\nValeurs réelles vs prédites (Réseau de neurones) :\n")
for real, pred in zip(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred_mlp)):
    results_file.write(f"Réel : {real}, Prédit : {pred}\n")
results_file.write("\n" + "="*50 + "\n")

# Matrice de confusion pour MLP
conf_mat_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure()
sns.heatmap(conf_mat_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matrice de confusion : Réseau de neurones (MLP)")
plt.xlabel("Valeurs prédites")
plt.ylabel("Valeurs réelles")
plt.savefig("matrice_confusion_mlp.png")
plt.close()

# Export des valeurs réelles et prédites pour MLP
comparison_mlp = pd.DataFrame({
    'Valeurs réelles': label_encoder.inverse_transform(y_test),
    'Valeurs prédites': label_encoder.inverse_transform(y_pred_mlp)
})
comparison_mlp.to_csv("reel_vs_predit_mlp.csv", index=False)

# Courbe ROC pour MLP
fpr, tpr, _ = roc_curve(y_test, proba_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (MLP)')
plt.legend(loc="lower right")
plt.savefig("roc_curve_mlp.png")
plt.close()

# Courbe PR pour MLP
precision, recall, _ = precision_recall_curve(y_test, proba_mlp[:, 1], pos_label=1)
plt.figure()
plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (MLP)')
plt.legend(loc="lower left")
plt.savefig("precision_recall_curve_mlp.png")
plt.close()

# Histogramme des erreurs pour MLP
errors = label_encoder.inverse_transform(y_test) != label_encoder.inverse_transform(y_pred_mlp)
plt.figure()
sns.histplot(errors, kde=False, bins=2)
plt.title('Histogramme des Erreurs (MLP)')
plt.xlabel('Erreur (1 = erreur)')
plt.ylabel('Nombre de prédictions')
plt.savefig("histogramme_erreurs_mlp.png")
plt.close()

results_file.close()

# Matrice de corrélation (nb_tokens vs category_nps après encodage)
df_clean['category_nps_encoded'] = label_encoder.transform(df_clean['category_nps'])
correlation_matrix = df_clean[['nb_tokens', 'category_nps_encoded']].corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matrice de corrélation")
plt.savefig("matrice_correlation.png")
plt.close()
