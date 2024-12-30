# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import f_oneway

# Création des dossiers pour les outputs
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
output_graph_folder = os.path.join(output_folder, 'graphs')
os.makedirs(output_graph_folder, exist_ok=True)
output_insight_file = os.path.join(output_folder, 'learned_insights.txt')

# ===== ANALYSE EXPLORATOIRE DU PREMIER DATAFRAME =====
def analyse_exploratoire_initiale(file_path):
    df = pd.read_csv(file_path)

    # Informations principales sur le DataFrame
    apercu = f"""
    --- ANALYSE EXPLORATOIRE DU PREMIER DATAFRAME ---
    Nombre de lignes : {df.shape[0]}
    Nombre de colonnes : {df.shape[1]}
    Colonnes disponibles : {list(df.columns)}
    Types de données :\n{df.dtypes}
    Valeurs nulles par colonne :\n{df.isnull().sum()}
    Duplicats : {df.duplicated().sum()} lignes dupliquées
    """
    print(apercu)

    # Sauvegarde dans le fichier texte
    with open(output_insight_file, 'w') as f:
        f.write(apercu)

    # Graphique 1 : Répartition des commentaires par plateforme
    plt.figure(figsize=(8, 6))
    df['plate_forme'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Graphique 1 : Répartition des commentaires par plateforme')
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_folder, 'graph_1_platform_distribution.png'))
    plt.close()

    # Graphique 2 : Distribution des Notes sans décimales
    if 'rating' in df.columns:
        plt.figure(figsize=(8, 6))
        df['rating'].hist(bins=5, color='green', alpha=0.7)
        plt.title('Graphique 2 : Distribution des Notes')
        plt.xlabel('Notes')
        plt.ylabel('Fréquence')
        plt.xticks(range(int(df['rating'].min()), int(df['rating'].max()) + 1))  # Forcer valeurs entières
        plt.tight_layout()
        plt.savefig(os.path.join(output_graph_folder, 'graph_2_distribution_ratings.png'))
        plt.close()

    # Graphique 3 : Nombre de commentaires par mois et par plateforme
    df['month_year'] = pd.to_datetime(df['month_year'])
    comments_by_time_platform = df.groupby(['month_year', 'plate_forme']).size().unstack(fill_value=0)
    comments_by_time_platform.plot(kind='bar', stacked=True, figsize=(12, 7))
    plt.title('Graphique 3 : Nombre de commentaires par mois et par plateforme')
    plt.xlabel('Temps (Mois)')
    plt.ylabel('Nombre de commentaires')
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_folder, 'graph_3_comments_by_month_platform.png'))
    plt.close()

# ===== ANALYSE EXPLORATOIRE AVANCÉE DU SECOND DATAFRAME =====
def analyse_exploratoire_avancee(file_path):
    df = pd.read_csv(file_path)

    # Graphique 4 : Médiane et Moyenne des Tokens par Segment NPS
    nps_tokens_stats = df.groupby('category_nps')['nb_tokens'].agg(['median', 'mean'])
    nps_tokens_stats.plot(kind='bar', figsize=(8, 6), color=['orange', 'blue'])
    plt.title('Graphique 4 : Médiane et Moyenne des Tokens par Segment NPS')
    plt.xlabel('Segment NPS')
    plt.ylabel('Nombre de Tokens')
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_folder, 'graph_4_tokens_stats_nps.png'))
    plt.close()

    # Graphique 5 : Moyenne des Tokens par Plateforme et Segment NPS
    avg_tokens_by_platform_nps = df.groupby(['plate_forme', 'category_nps'])['nb_tokens'].mean().unstack()
    avg_tokens_by_platform_nps.plot(kind='bar', figsize=(10, 6))
    plt.title('Graphique 5 : Moyenne des Tokens par Plateforme et Segment NPS')
    plt.xlabel('Plateforme')
    plt.ylabel('Nombre moyen de Tokens')
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_folder, 'graph_5_avg_tokens_platform_nps.png'))
    plt.close()

# ===== TEST ANOVA =====
def test_anova(file_path):
    df = pd.read_csv(file_path)
    print("----- Test ANOVA -----")
    groups = [df[df['category_nps'] == cat]['nb_tokens'] for cat in df['category_nps'].unique()]
    stat, p_value = f_oneway(*groups)
    print(f"Statistique F : {stat:.4f}, p-value : {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion : Rejet de H0 - Les différences entre les groupes sont significatives.")
    else:
        print("Conclusion : On ne rejette pas H0 - Aucune différence significative.")

    # Graphique : Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='category_nps', y='nb_tokens', palette='muted')
    plt.title('Boxplot des longueurs de commentaires par Segment NPS')
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_folder, 'boxplot_anova_tokens_nps.png'))
    plt.close()

    # Graphique : Barplot avec intervalles de confiance
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='category_nps', y='nb_tokens', ci='sd', palette='muted')
    plt.title('Moyenne des Tokens avec Intervalle de Confiance par Segment NPS')
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_folder, 'barplot_anova_tokens_nps.png'))
    plt.close()

# ===== EXECUTION DU SCRIPT =====
file_path_eda1 = 'df_09.csv'
file_path_eda2 = 'df_cleaned_with_analysis_df_09_avec_category_nps.csv'

print("\n--- ETAPE 1 : ANALYSE EXPLORATOIRE INITIAL ---")
analyse_exploratoire_initiale(file_path_eda1)

print("\n--- ETAPE 2 : ANALYSE EXPLORATOIRE AVANCÉE ---")
analyse_exploratoire_avancee(file_path_eda2)

print("\n--- ETAPE 3 : TEST ANOVA ---")
test_anova(file_path_eda2)
