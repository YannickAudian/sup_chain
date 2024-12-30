import streamlit as st
import pandas as pd
import os
import plotly.express as px
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Ignorer les warnings inutiles
import warnings
warnings.filterwarnings('ignore')

# ===== Sommaire interactif =====
st.sidebar.title("Sommaire")
page = st.sidebar.radio(
    "Navigation",
    ["Page 1 : EDA complète", "Page 2 : Modélisation (Tokens vs Sentiment)"]
)

# ===== Configuration des dossiers pour sauvegarde =====
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
output_graph_folder = os.path.join(output_folder, 'graphs')
os.makedirs(output_graph_folder, exist_ok=True)

# ===== Fichiers de données =====
file_path_eda1 = 'df_09.csv'
file_path_eda2 = 'df_cleaned_with_analysis_df_09_avec_category_nps.csv'

# ===== Page 1 : EDA complète =====
if page == "Page 1 : EDA complète":
    st.title("Exploration des Données (EDA complète)")

    # ===== Section 1 : Analyse Exploratoire Initiale =====
    st.header("1. Analyse Exploratoire Initiale")
    df1 = pd.read_csv(file_path_eda1)

    # Suppression de la colonne 'category' si présente
    if 'category' in df1.columns:
        df1.drop(columns=['category'], inplace=True)

    st.subheader("Informations générales")
    st.write(f"**Nombre de lignes :** {df1.shape[0]}")
    st.write(f"**Nombre de colonnes :** {df1.shape[1]}")
    st.write("**Colonnes disponibles :**")
    st.dataframe(pd.DataFrame({'Variable': df1.columns}))

    st.write("**Valeurs nulles par colonne :**")
    st.dataframe(df1.isnull().sum().reset_index().rename(columns={'index': 'Variable', 0: 'Valeurs nulles'}))

    # Graphique 1 : Répartition des commentaires par plateforme
    if 'plate_forme' in df1.columns:
        st.write("### Graphique 1 : Répartition des commentaires par plateforme")
        fig = px.pie(
            df1, 
            names='plate_forme', 
            title="Répartition des commentaires par plateforme",
            labels={'plate_forme': 'Plateforme'}
        )
        st.plotly_chart(fig)

    # Graphique 2 : Distribution des Notes
    if 'rating' in df1.columns:
        st.write("### Graphique 2 : Distribution des Notes")
        fig = px.histogram(
            df1, 
            x='rating', 
            nbins=5, 
            title="Distribution des Notes",
            labels={'rating': 'Note'}
        )
        fig.update_layout(
            xaxis_title="Notes",
            yaxis_title="Nombre de commentaires"
        )
        st.plotly_chart(fig)

    # Graphique 3 : Nombre de commentaires par année et plateforme
    if 'month_year' in df1.columns and 'plate_forme' in df1.columns:
        df1['month_year'] = pd.to_datetime(df1['month_year'])
        df1['year'] = df1['month_year'].dt.year  # Extraire l'année
        comments_by_year_platform = df1.groupby(['year', 'plate_forme']).size().reset_index(name='count')
        st.write("### Graphique 3 : Nombre de commentaires par année et plateforme")
        fig = px.bar(
            comments_by_year_platform, 
            x='year', 
            y='count', 
            color='plate_forme',
            title="Nombre de commentaires par année et par plateforme",
            labels={
                'year': 'Année',
                'count': 'Nombre de commentaires',
                'plate_forme': 'Plateforme'
            }
        )
        fig.update_layout(
            xaxis_title="Année",
            yaxis_title="Nombre de commentaires"
        )
        st.plotly_chart(fig)

    # ===== Section 2 : Analyse Exploratoire Avancée =====
    st.header("2. Analyse Exploratoire Avancée")
    df2 = pd.read_csv(file_path_eda2)

    st.subheader("Description du deuxième dataset")
    st.write(f"**Nombre de lignes :** {df2.shape[0]}")
    st.write(f"**Nombre de colonnes :** {df2.shape[1]}")
    st.write("**Colonnes disponibles :**")
    st.dataframe(pd.DataFrame({'Variable': df2.columns}))

    st.write("**Valeurs nulles par colonne :**")
    st.dataframe(df2.isnull().sum().reset_index().rename(columns={'index': 'Variable', 0: 'Valeurs nulles'}))

    # Graphique 4 : Médiane et Moyenne des Tokens par Segment NPS
    if 'category_nps' in df2.columns and 'nb_tokens' in df2.columns:
        nps_tokens_stats = df2.groupby('category_nps')['nb_tokens'].agg(['median', 'mean']).reset_index()
        st.write("### Graphique 4 : Médiane et Moyenne des Tokens par Segment NPS")
        fig = px.bar(
            nps_tokens_stats, 
            x='category_nps', 
            y=['median', 'mean'], 
            barmode='group',
            title="Médiane et Moyenne des Tokens par Segment NPS",
            labels={
                'category_nps': 'Segment NPS',
                'value': 'Valeur',
                'variable': 'Statistique'
            }
        )
        fig.update_layout(
            xaxis_title="Segment NPS",
            yaxis_title="Valeur"
        )
        st.plotly_chart(fig)

    # ===== Section 3 : Test ANOVA =====
    st.header("3. Test ANOVA")
    if 'category_nps' in df2.columns and 'nb_tokens' in df2.columns:
        groups = [df2[df2['category_nps'] == cat]['nb_tokens'] for cat in df2['category_nps'].unique()]
        stat, p_value = f_oneway(*groups)
        st.write(f"**Statistique F :** {stat:.4f}")
        st.write(f"**p-value :** {p_value:.4f}")

        if p_value < 0.05:
            st.write("**Conclusion :** Rejet de H0 - Les différences entre les groupes sont significatives.")
        else:
            st.write("**Conclusion :** On ne rejette pas H0 - Aucune différence significative.")

        # Graphique : Boxplot
        st.write("### Boxplot des longueurs de commentaires par Segment NPS")
        fig = px.box(
            df2, 
            x='category_nps', 
            y='nb_tokens', 
            title="Boxplot des longueurs de commentaires par Segment NPS",
            labels={
                'category_nps': 'Segment NPS',
                'nb_tokens': 'Nombre de Tokens'
            }
        )
        st.plotly_chart(fig)

# ===== Page 2 : Modélisation =====
elif page == "Page 2 : Modélisation (Tokens vs Sentiment)":
    st.title("Modélisation : Tokens vs Sentiment")

    # Chargement et prétraitement des données
    df = pd.read_csv(file_path_eda2)
    df_clean = df[['category_nps', 'nb_tokens']].dropna()

    def reassign_neutral(row):
        if row['category_nps'] == 'Neutre':
            if row['nb_tokens'] < 3:
                return 'Détracteur'
            else:
                return 'Promoteur'
        return row['category_nps']

    df_clean['category_nps'] = df_clean.apply(reassign_neutral, axis=1)

    label_encoder = LabelEncoder()
    df_clean['category_nps_encoded'] = label_encoder.fit_transform(df_clean['category_nps'])

    X = df_clean[['nb_tokens']].values
    y = df_clean['category_nps_encoded'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Sélection du modèle
    models = {
        'LogisticRegression': LogisticRegression(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'SVM': SVC(),
        'XGBoost': XGBClassifier(eval_metric='logloss'),
    }

    model_choice = st.selectbox("Choisissez un modèle :", list(models.keys()))

    if st.button("Lancer la Modélisation"):
        model = models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métriques
        st.subheader("Résultats")
        st.write(f"**Modèle :** {model_choice}")
        st.write(f"**Accuracy :** {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"**F1 Score :** {f1_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"**Precision :** {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"**Recall :** {recall_score(y_test, y_pred, average='weighted'):.2f}")

        # Matrice de confusion
        st.subheader("Matrice de Confusion")
        confusion = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        ax.set_title("Matrice de Confusion")
        st.pyplot(fig)
