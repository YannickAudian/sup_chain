Analyse de Classification avec Comparaison de Modèles et Réseau de Neurones
Objectif du Projet
L'objectif de ce projet est de classer les données en fonction de la catégorie NPS (Promoteur, Détracteur, Neutre) en utilisant plusieurs algorithmes de classification. Le script compare les performances de divers modèles, dont un réseau de neurones (MLP), en utilisant des métriques telles que l'accuracy, les courbes ROC et Precision-Recall, ainsi que les matrices de confusion. Ce projet permet également d'identifier les meilleures configurations pour chaque modèle grâce à une recherche d'hyperparamètres (GridSearchCV).

Fonctionnalités
Chargement et Prétraitement des Données :

Les données sont chargées depuis un fichier CSV nommé df_09_avec_category_nps.csv.
Les valeurs manquantes sont supprimées, et une réaffectation des catégories NPS est réalisée (basée sur le nombre de tokens dans chaque observation).
Encodage des Données :

Les labels sont encodés pour être compatibles avec les modèles de classification.
Comparaison de Modèles :

Le script teste plusieurs modèles de classification :
Régression Logistique
Arbre de Décision
Forêt Aléatoire
Gradient Boosting
Support Vector Machines (SVM)
XGBoost
Les modèles sont comparés sur la base de leur précision et d'autres métriques clés.
Réseau de Neurones (MLP) :

Un perceptron multicouche (MLPClassifier) est entraîné pour la classification.
Le réseau est évalué à l'aide de courbes ROC, Precision-Recall, et d'un histogramme des erreurs.
Génération de Rapports et Graphiques :

Les résultats sont sauvegardés dans un fichier texte resultats.txt.
Des graphiques sont générés pour visualiser les performances :
Matrices de confusion
Courbes ROC et Precision-Recall
Histogramme des erreurs
Prérequis
Bibliothèques Python
Le projet utilise les bibliothèques suivantes :

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
Installation des dépendances
Pour installer les bibliothèques nécessaires, exécutez la commande suivante :

bash
Copier le code
pip install -r requirements.txt
Structure des Fichiers
Script Principal
Le script principal est le fichier .py qui contient tout le code.

Fichier CSV
Les données sont contenues dans un fichier nommé df_09_avec_category_nps.csv. Ce fichier doit contenir les colonnes suivantes :

category_nps : La catégorie NPS initiale (Promoteur, Détracteur, Neutre).
nb_tokens : Le nombre de tokens associé à chaque observation.
Fichiers de Sortie
Fichier Texte :

resultats.txt : Contient les métriques des modèles, les valeurs réelles vs prédites, et les résultats de classification.
Graphiques :

matrice_confusion_<nom_du_modele>.png : Matrices de confusion pour chaque modèle.
roc_curve_mlp.png : Courbe ROC pour le réseau de neurones.
precision_recall_curve_mlp.png : Courbe Precision-Recall pour le réseau de neurones.
histogramme_erreurs_mlp.png : Histogramme des erreurs pour le réseau de neurones.
matrice_correlation.png : Matrice de corrélation des données.
Valeurs Réelles vs Prédites :

reel_vs_predit_<nom_du_modele>.csv : Fichiers contenant les prédictions pour chaque modèle.
Exécution
Étapes pour Exécuter le Script
Préparation :

Placez le fichier df_09_avec_category_nps.csv dans le même répertoire que le script.
Exécution :

Lancez le script Python :
bash
Copier le code
python nom_du_script.py
Résultats :

Consultez les résultats dans le fichier resultats.txt.
Visualisez les performances des modèles à l'aide des graphiques générés.
Résultats Attendus
Comparaison des Modèles :

Précision des différents modèles sur l'ensemble de test.
Rapport de classification détaillé pour chaque modèle.
Visualisations :

Matrices de confusion pour analyser les erreurs de classification.
Courbes ROC et Precision-Recall pour évaluer les performances des modèles probabilistes.
Histogramme des erreurs pour le réseau de neurones.
Export des Prédictions :

Les valeurs réelles vs prédites sont sauvegardées dans des fichiers CSV pour une analyse plus approfondie.
Améliorations Possibles
Augmentation des Données :

Ajout de plus de caractéristiques pour améliorer la précision des modèles.
Optimisation des Hyperparamètres :

Étendre la recherche des hyperparamètres pour chaque modèle.
Ajout d'autres Algorithmes :

Tester des modèles comme LightGBM ou des réseaux convolutifs (CNN) pour des données plus complexes.
Auteur
Ce projet a été développé pour analyser et comparer les performances de plusieurs algorithmes de classification sur un ensemble de données NPS.

Pour toute question ou suggestion, n'hésitez pas à me contacter !