# ===== Importation des Bibliothèques =====
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ===== Configuration des Dossiers =====
output_folder = 'int_output'
os.makedirs(output_folder, exist_ok=True)
graphs_folder = os.path.join(output_folder, 'graphs')
os.makedirs(graphs_folder, exist_ok=True)

# ===== Chargement des Données =====
df = pd.read_csv('df_cleaned_with_analysis_df_09_avec_category_nps.csv')

# ===== Prétraitement des Données =====
df_clean = df[['category_nps', 'nb_tokens']].dropna()
df_clean['category_nps_encoded'] = df_clean['category_nps'].factorize()[0]

# Séparation des données
X = df_clean[['nb_tokens']].values
y = df_clean['category_nps_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ===== Chargement du Meilleur Modèle =====
# Remplacez ce modèle par le meilleur modèle trouvé dans votre script principal
from sklearn.ensemble import GradientBoostingClassifier
best_model = GradientBoostingClassifier().fit(X_train, y_train)

# ===== LIME pour l'Interprétabilité =====
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=['nb_tokens'],
    class_names=['Détracteur', 'Promoteur', 'Neutre'],
    discretize_continuous=True,
    random_state=42
)

# Sélection d'un échantillon pour interprétation
test_instance = X_test[0].reshape(1, -1)
lime_exp = explainer.explain_instance(
    test_instance[0], 
    best_model.predict_proba, 
    num_features=1
)

# Sauvegarde des Explications
lime_exp.save_to_file(os.path.join(output_folder, 'lime_explanation.html'))

# ===== Sauvegarde des Métriques =====
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['Détracteur', 'Promoteur', 'Neutre'])

with open(os.path.join(output_folder, 'metrics_results.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep)

# ===== Matrice de Confusion =====
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Détracteur', 'Promoteur', 'Neutre'], yticklabels=['Détracteur', 'Promoteur', 'Neutre'])
plt.title("Matrice de Confusion")
plt.xlabel("Prédiction")
plt.ylabel("Vérité")
plt.tight_layout()
plt.savefig(os.path.join(graphs_folder, 'confusion_matrix.png'))
plt.close()

print("Le script est terminé. Les résultats sont sauvegardés dans le dossier 'int_output'.")
