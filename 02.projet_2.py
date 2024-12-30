import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Importation des algorithmes de classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Pour le réseau de neurones
from sklearn.neural_network import MLPClassifier

# Pour ignorer les warnings
import warnings
warnings.filterwarnings('ignore')

# 1. Chargement des données
df = pd.read_csv('df_09_avec_category_nps.csv')

# 2. Prétraitement des données
# Suppression des valeurs manquantes
df_clean = df[['category_nps', 'nb_tokens']].dropna()

# Réaffectation de la classe 'Neutre' selon la règle spécifiée
def reassign_neutral(row):
    if row['category_nps'] == 'Neutre':
        # Selon votre règle : moins de 3 => Détracteur, 3 ou plus => Promoteur
        if row['nb_tokens'] < 3:
            return 'Détracteur'
        else:
            return 'Promoteur'
    else:
        return row['category_nps']

df_clean['category_nps'] = df_clean.apply(reassign_neutral, axis=1)

# Vérification de la distribution des classes
print("Distribution des classes après réaffectation :")
print(df_clean['category_nps'].value_counts(normalize=True) * 100)

# 3. Préparation des données pour la modélisation
X = df_clean[['nb_tokens']].values
y = df_clean['category_nps'].values

# Encodage des labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Séparation des données en ensembles d'entraînement et de test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# 5. Définition des modèles et des hyperparamètres à tester
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
}

param_grids = {
    'LogisticRegression': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear'],
    },
    'DecisionTree': {
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
    },
}

# 6. Création d'un pipeline pour la modélisation
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

# 7. Recherche du meilleur modèle
best_model = None
best_score = 0
best_model_name = ''
best_params = {}

for model_name in models.keys():
    print(f"\nTraitement du modèle : {model_name}")
    pipeline = Pipeline([
        ('transformer', IdentityTransformer()),  # Placeholder si vous ajoutez des transformations plus tard
        ('classifier', models[model_name])
    ])
    param_grid = param_grids[model_name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model_score = grid_search.best_score_
    print(f"Meilleure précision en validation croisée pour {model_name}: {model_score:.4f}")
    print(f"Meilleurs hyperparamètres pour {model_name}: {grid_search.best_params_}")
    if model_score > best_score:
        best_score = model_score
        best_model = grid_search.best_estimator_
        best_model_name = model_name
        best_params = grid_search.best_params_

print(f"\nMeilleur modèle sélectionné : {best_model_name} avec une précision de {best_score:.4f}")

# 8. Évaluation du meilleur modèle sur l'ensemble de test
y_pred = best_model.predict(X_test)

# Calcul de la précision globale
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nPrécision globale sur l'ensemble de test : {accuracy:.2f}%")

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Calcul du pourcentage de bonnes prédictions par classe
conf_mat = confusion_matrix(y_test, y_pred)
for idx, category in enumerate(label_encoder.classes_):
    true_positive = conf_mat[idx, idx]
    total_actual = conf_mat[idx, :].sum()
    class_accuracy = (true_positive / total_actual) * 100
    print(f"Pourcentage de bonnes prédictions pour la classe '{category}' : {class_accuracy:.2f}%")

# Distribution réelle des classes dans l'ensemble de test
real_distribution = pd.Series(y_test).value_counts(normalize=True) * 100
print("\nPourcentage de chaque catégorie dans les données réelles :")
for idx, percentage in real_distribution.items():
    print(f"{label_encoder.classes_[idx]} : {percentage:.2f}%")

# Distribution des prédictions
predicted_distribution = pd.Series(y_pred).value_counts(normalize=True) * 100
print("\nPourcentage de chaque catégorie dans les prédictions :")
for idx, percentage in predicted_distribution.items():
    print(f"{label_encoder.classes_[idx]} : {percentage:.2f}%")

# 9. Entraînement d'un réseau de neurones
# Affiche un message pour indiquer le début de l'entraînement du réseau de neurones
print("\n--- Entraînement du réseau de neurones ---")

# Création d'un pipeline pour le réseau de neurones
mlp_pipeline = Pipeline([
    ('transformer', IdentityTransformer()),  # Étape de transformation (ne fait rien ici, mais permet d'ajouter des transformations futures si nécessaire)
    ('classifier', MLPClassifier(max_iter=500, random_state=42))  # Classificateur : Multi-Layer Perceptron (réseau de neurones), avec un nombre maximal d'itérations de 500 et une graine aléatoire pour reproductibilité
])

# Définition de la grille des hyperparamètres à tester pour MLPClassifier
mlp_param_grid = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Structure des couches cachées : 
                                                                  # - (50,) : Une seule couche cachée avec 50 neurones
                                                                  # - (100,) : Une seule couche cachée avec 100 neurones
                                                                  # - (50, 50) : Deux couches cachées avec 50 neurones chacune

    'classifier__activation': ['tanh', 'relu'],  # Fonction d'activation :
                                                  # - 'tanh' : Fonction tangente hyperbolique
                                                  # - 'relu' : Rectified Linear Unit (ReLU), souvent utilisée pour les réseaux de neurones profonds

    'classifier__solver': ['adam'],  # Algorithme d'optimisation :
                                      # - 'adam' : Méthode de descente de gradient adaptative très efficace pour de nombreux problèmes

    'classifier__alpha': [0.0001, 0.001],  # Paramètre de régularisation L2 :
                                            # - Permet d'éviter le sur-apprentissage en pénalisant les poids trop élevés

    'classifier__learning_rate': ['constant', 'adaptive'],  # Stratégie de taux d'apprentissage :
                                                            # - 'constant' : Taux d'apprentissage fixe
                                                            # - 'adaptive' : Le taux d'apprentissage diminue lorsque l'entraînement n'améliore pas les performances
}

# Recherche du meilleur modèle avec GridSearchCV
mlp_grid_search = GridSearchCV(
    mlp_pipeline,                # Pipeline avec le transformateur et le classificateur
    mlp_param_grid,              # Grille des hyperparamètres à tester
    cv=cv,                       # Validation croisée stratifiée à 5 plis (définie précédemment)
    scoring='accuracy',          # Métrique d'évaluation : précision
    n_jobs=-1                    # Utilisation de tous les cœurs disponibles pour paralléliser les calculs
)

# Entraînement du GridSearch sur les données d'entraînement
mlp_grid_search.fit(X_train, y_train)

# Récupération du meilleur modèle trouvé par GridSearch
mlp_best_model = mlp_grid_search.best_estimator_

# Récupération du meilleur score obtenu en validation croisée
mlp_best_score = mlp_grid_search.best_score_

# Affichage des résultats de la recherche
print(f"Meilleure précision en validation croisée pour MLPClassifier: {mlp_best_score:.4f}")
print(f"Meilleurs hyperparamètres pour MLPClassifier: {mlp_grid_search.best_params_}")

# Évaluation du réseau de neurones sur l'ensemble de test
y_pred_mlp = mlp_best_model.predict(X_test)

# Calcul de la précision globale
accuracy_mlp = accuracy_score(y_test, y_pred_mlp) * 100
print(f"\nPrécision globale du réseau de neurones sur l'ensemble de test : {accuracy_mlp:.2f}%")

# Rapport de classification
print("\nRapport de classification (Réseau de neurones) :")
print(classification_report(y_test, y_pred_mlp, target_names=label_encoder.classes_))

# Calcul du pourcentage de bonnes prédictions par classe
conf_mat_mlp = confusion_matrix(y_test, y_pred_mlp)
for idx, category in enumerate(label_encoder.classes_):
    true_positive = conf_mat_mlp[idx, idx]
    total_actual = conf_mat_mlp[idx, :].sum()
    class_accuracy = (true_positive / total_actual) * 100
    print(f"Pourcentage de bonnes prédictions pour la classe '{category}' (Réseau de neurones) : {class_accuracy:.2f}%")

# Distribution des prédictions
predicted_distribution_mlp = pd.Series(y_pred_mlp).value_counts(normalize=True) * 100
print("\nPourcentage de chaque catégorie dans les prédictions (Réseau de neurones) :")
for idx, percentage in predicted_distribution_mlp.items():
    print(f"{label_encoder.classes_[idx]} : {percentage:.2f}%")