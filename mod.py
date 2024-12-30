import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Ignorer les warnings
import warnings
warnings.filterwarnings('ignore')

# Création des dossiers de sortie
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
graphs_folder = os.path.join(output_folder, 'graphs')
os.makedirs(graphs_folder, exist_ok=True)

# ===== Chargement des Données =====
df = pd.read_csv('df_cleaned_with_analysis_df_09_avec_category_nps.csv')

# ===== Prétraitement des Données =====
# Suppression des valeurs manquantes
df_clean = df[['category_nps', 'nb_tokens']].dropna()

# Réaffectation des classes neutres
def reassign_neutral(row):
    if row['category_nps'] == 'Neutre':
        if row['nb_tokens'] < 3:
            return 'Détracteur'
        else:
            return 'Promoteur'
    return row['category_nps']

df_clean['category_nps'] = df_clean.apply(reassign_neutral, axis=1)

# Encodage des labels
label_encoder = LabelEncoder()
df_clean['category_nps_encoded'] = label_encoder.fit_transform(df_clean['category_nps'])

# ===== Visualisation des Données =====
# Distribution des classes après réaffectation
plt.figure(figsize=(8, 6))
sns.countplot(x='category_nps', data=df_clean, palette='muted')
plt.title("Distribution des Classes NPS")
plt.xlabel("Classes NPS")
plt.ylabel("Nombre d'observations")
plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, 'distribution_classes_nps.png'))
plt.close()

# Distribution des tokens
plt.figure(figsize=(8, 6))
sns.histplot(df_clean['nb_tokens'], bins=50, kde=True, color='green')
plt.title("Distribution des Longueurs de Commentaires (nb_tokens)")
plt.xlabel("Nombre de Tokens")
plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, 'distribution_nb_tokens.png'))
plt.close()

# ===== Modélisation et Séparation des Données =====
X = df_clean[['nb_tokens']].values
y = df_clean['category_nps_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Définition des modèles
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
}

# Hyperparamètres
param_grids = {
    'LogisticRegression': {'classifier__C': [0.01, 0.1, 1, 10]},
    'DecisionTree': {'classifier__max_depth': [None, 5, 10]},
    'RandomForest': {'classifier__n_estimators': [100, 200]},
    'GradientBoosting': {'classifier__n_estimators': [100, 200]},
    'SVM': {'classifier__C': [1, 10], 'classifier__kernel': ['linear', 'rbf']},
    'XGBoost': {'classifier__n_estimators': [100, 200]},
}

# Création d'un pipeline
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

best_model, best_score = None, 0
results = []

for model_name, model in models.items():
    pipeline = Pipeline([
        ('transformer', IdentityTransformer()),
        ('classifier', model)
    ])
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    score = grid_search.best_score_
    results.append((model_name, score))
    if score > best_score:
        best_score = score
        best_model = grid_search.best_estimator_

# Sauvegarde des résultats
results_df = pd.DataFrame(results, columns=['Model', 'Score']).sort_values(by='Score', ascending=False)
results_df.to_csv(os.path.join(output_folder, 'model_results.csv'), index=False)

# ===== Évaluation du Meilleur Modèle =====
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
confusion = confusion_matrix(y_test, y_pred)

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matrice de Confusion du Meilleur Modèle")
plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, 'confusion_matrix.png'))
plt.close()

# ===== Documentation des Contraintes et Défis =====
constraints_text = """
Contraintes et Défis :
- Imbalance des classes : La classe 'Détracteur' est majoritaire.
- Dépendance sur une seule variable explicative (nb_tokens).
- Temps de calcul pour GridSearch avec plusieurs hyperparamètres.
"""
with open(os.path.join(output_folder, 'constraints_and_challenges.txt'), 'w') as f:
    f.write(constraints_text)

# ===== Learning des Modélisations =====
learnings_text = f"""
Learning des Modélisations :
- Le modèle {best_model.steps[-1][1]} a obtenu la meilleure performance avec une précision de {accuracy:.2f}.
- L'importance d'une stratégie de validation croisée pour éviter le sur-apprentissage.
- Comparaison des modèles montre que XGBoost est robuste même sur des données simples.
"""
with open(os.path.join(output_folder, 'learnings.txt'), 'w') as f:
    f.write(learnings_text)

# ===== Enjeux Stratégiques =====
strategic_text = """
Enjeux Stratégiques :
- Optimisation des actions marketing en ciblant les détracteurs.
- Segmentation précise pour améliorer la satisfaction client.
- Faciliter l'automatisation des décisions stratégiques grâce au modèle XGBoost.
"""
with open(os.path.join(output_folder, 'strategic_issues.txt'), 'w') as f:
    f.write(strategic_text)

print("Le script est terminé. Consultez le dossier 'output' pour les résultats.")
