# PREPARATION DE LA DATA
# INPUTS: df_09_avec_category_nps.csv (travaillé par Yann et Nono)
# TRAITEMENT: on utilise spacy pour extraire les catégories gramaticales des
# reviews des consommateurs
# OUTPUT: pandas dataframe contenant en incluant comme colonnes les nombre des différentes
# catégories grammaticales, heatmap (corrélations) entre les variables et plots interactifs
# caractérisant les consommateurs détracteurs, les neutres et le promoteurs



# ATTENTION: Il faut personnaliser les PATHS avant d'exécuter ce fichier

# lecture des données
try:
    import sys
    PACKAGES_PATH = 'C:\\Bib\\WPy64-3880\\notebooks\\docs\\9999_projet_fil_rouge\\pythonProject\\.venv\\Lib\\site-packages\\'
    sys.path.insert(1, PACKAGES_PATH)
    import pandas as pd
    import numpy as np
    import spacy
    from collections import Counter
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder

    # fonctions auxiliaires:

    def plot_distributions(category_name, category_df, variables_a, variables_b):
        fig = make_subplots(rows=2, cols=4,
                            subplot_titles=(variables_a[0], variables_a[1], variables_a[2], variables_a[3],
                                            variables_b[0], variables_b[1], variables_b[2], variables_b[3]))
        for ii in range(len(variables_a)):
            fig.add_trace(go.Histogram(x=category_df[variables_a[ii]]),
                          row=1, col=ii + 1)
            mediane = category_df[variables_a[ii]].median()

            fig.add_vline(x=mediane,
                          line_color="chartreuse",
                          line_dash="dot",
                          annotation_text=str(round(mediane, 2)),
                          annotation_position="top right", annotation_font_color="chartreuse",
                          row=1, col=ii + 1)
        for ii in range(len(variables_b)):
            fig.add_trace(go.Histogram(x=category_df[variables_b[ii]]),
                          row=2, col=ii + 1)

            mediane = category_df[variables_b[ii]].median()
            fig.add_vline(x=mediane,
                          line_color="chartreuse",
                          line_dash="dot",
                          annotation_text=str(round(mediane, 2)),
                          annotation_position="top right",
                          annotation_font_color="chartreuse",
                          row=2, col=ii + 1)
        fig.update_layout(title_text=category_name, title_x=0.5)
        fig.update_layout(showlegend=False)
        fig.write_html(DATA_PATH + category_name +".html")
        fig.show()
    def count_categories(doc):
        '''
        Count the number of 'ADJ', 'NOUN', 'PRON', 'PROPN', 'ADV', 'VERB', 'INTJ', 'SYM'
        in doc (type doc de spacy)
        :param doc:
        :return a list per category (type)
        '''
        adjs = list()
        nouns = list()
        pronouns = list()
        prop_nouns = list()
        adverbs = list()
        verbs = list()
        interjections = list()
        symbols = list()
        adjs_tally = 0
        nouns_tally = 0
        pronouns_tally = 0
        prop_nouns_tally = 0
        adverbs_tally = 0
        verbs_tally = 0
        interjections_tally = 0
        symbols_tally = 0

        for token in doc:
            if token.pos_ == 'ADJ':
                adjs.append(token.text)
            elif token.pos_ == 'NOUN':
                nouns.append(token.text)
            elif token.pos_ == 'PRON':
                pronouns.append(token.text)
            elif token.pos_ == 'PROPN':
                prop_nouns.append(token.text)
            elif token.pos_ == 'ADV':
                adverbs.append(token.text)
            elif token.pos_ == 'VERB':
                verbs.append(token.text)
            elif token.pos_ == 'INTJ':
                interjections.append(token.text)
            elif token.pos_ == 'SYM':
                symbols.append(token.text)

            adjs_tally = len(Counter(adjs))
            nouns_tally = len(Counter(nouns))
            pronouns_tally = len(Counter(pronouns))
            prop_nouns_tally = len(Counter(prop_nouns))
            adverbs_tally = len(Counter(adverbs))
            verbs_tally = len(Counter(verbs))
            interjections_tally = len(Counter(interjections))
            symbols_tally = len(Counter(symbols))

        return (adjs, adjs_tally, nouns, nouns_tally, pronouns, pronouns_tally, prop_nouns, prop_nouns_tally, adverbs,
                adverbs_tally, verbs, verbs_tally, interjections, interjections_tally, symbols, symbols_tally)


    def complete_cols_categories(df):
        L_adjs = list()
        L_adjs_tally = list()
        L_nouns = list()
        L_nouns_tally = list()
        L_pronouns = list()
        L_pronouns_tally = list()
        L_prop_nouns = list()
        L_prop_nouns_tally = list()
        L_adverbs = list()
        L_adverbs_tally = list()
        L_verbs = list()
        L_verbs_tally = list()
        L_interjections = list()
        L_interjections_tally = list()
        L_symbols = list()
        L_symbols_tally = list()
        for _, row in df.iterrows():
            result = count_categories(nlp(row["body"]))
            L_adjs.append(result[0])
            L_adjs_tally.append(result[1])
            L_nouns.append(result[2])
            L_nouns_tally.append(result[3])
            L_pronouns.append(result[4])
            L_pronouns_tally.append(result[5])
            L_prop_nouns.append(result[6])
            L_prop_nouns_tally.append(result[7])
            L_adverbs.append(result[8])
            L_adverbs_tally.append(result[9])
            L_verbs.append(result[10])
            L_verbs_tally.append(result[11])
            L_interjections.append(result[12])
            L_interjections_tally.append(result[13])
            L_symbols.append(result[14])
            L_symbols_tally.append(result[15])

        df["adjs"] = L_adjs
        df["adjs_counts"] = L_adjs_tally
        df["nouns"] = L_nouns
        df["nouns_counts"] = L_nouns_tally
        df["pronouns"] = L_pronouns
        df["pronouns_counts"] = L_pronouns_tally
        df["prop_nouns"] = L_prop_nouns
        df["prop_nouns_counts"] = L_prop_nouns_tally
        df["adverbs"] = L_adverbs
        df["adverbs_counts"] = L_adverbs_tally
        df["verbs"] = L_verbs
        df["verbs_counts"] = L_verbs_tally
        df["interjections"] = L_interjections
        df["interjections_counts"] = L_interjections_tally
        df["symbols"] = L_symbols
        df["symbols_counts"] = L_symbols_tally

        return df


    DATA_PATH = "C:\\Bib\\WPy64-3880\\notebooks\\docs\\9999_projet_fil_rouge\\pythonProject\\data\\"
    FILE_NAME = DATA_PATH + "df_09_avec_category_nps.csv"
    df_row = pd.read_csv(FILE_NAME)

    nlp = spacy.load("fr_core_news_sm") # version 'français' de spacy
    avis_series = df_row["body"].to_numpy()

    df_full = complete_cols_categories(df_row) #ajouter les catégories

    to_be_kept = ['plate_forme', 'rating', 'nb_tokens', 'category_nps', 'category_numeric', 'adjs_counts', 'nouns_counts', 'pronouns_counts',
                  'prop_nouns_counts', 'adverbs_counts', 'verbs_counts', 'interjections_counts', 'symbols_counts']
    df = df_full[to_be_kept].copy()

    le = LabelEncoder()
    label = le.fit_transform(df['plate_forme'])
    df['plate_forme'] = label
    # AMAZON:0, CDISCOUNT: 1, RAKUTEN: 2, RDC:3

    # si on veut enregistrer le nouveaux df avec la nouvelle info,
    # dans mon cas je l'enregistre pour éviter de repéter cet étape
    # à chaque fois:
    FILE_NAME = DATA_PATH + "data_numerique.csv"
    df.to_csv(FILE_NAME)

     ##### description des variables #####

    correl_variables = ['category_numeric', 'rating', 'nb_tokens', 'adjs_counts','nouns_counts',
                         'pronouns_counts', 'prop_nouns_counts', 'adverbs_counts', 'verbs_counts',
                         'interjections_counts','symbols_counts']
    df_heat = df[correl_variables].copy()
    # heatmap: correlations simples
    title = "Corrélations entre les variables"
    fig, ax = plt.subplots(figsize=(10, 8))
    matrix = np.triu(df_heat.corr())
    sns.heatmap(df_heat.corr(),
                annot=True,
                vmin=-1,
                vmax=1,
                center=0,
                cmap='coolwarm',
                mask=matrix,
                annot_kws={"fontsize": 8})
    ax.set_title(title)
    plt.show()


    target = df['category_numeric'].copy()
    # on enleve les variables qui ont une corrélation très faible avec le target:
    feats = df.drop(['category_numeric', 'interjections_counts', 'symbols_counts'], axis=1).copy()

    # analyse "Détracteurs"
    data = pd.concat([target, feats], axis=1)
    data_detract = data[data["category_numeric"] == -1].copy()
    data_neutre = data[data["category_numeric"] == 0].copy()
    data_promot = data[data["category_numeric"] == 1].copy()

    variables_a = ['rating', 'nb_tokens', 'adjs_counts', 'nouns_counts']
    variables_b = ['pronouns_counts', 'prop_nouns_counts', 'adverbs_counts', 'verbs_counts']
    categories = [data_detract, data_neutre, data_promot]
    categories_names = ['Detracteurs', 'Neutres', 'Promoteurs']

    for ii in range(len(categories)):
        plot_distributions(categories_names[ii], categories[ii], variables_a, variables_b)

    print("Le fichier a été exécuté correctement")
    print("Le fichier 'data_numerique.csv' et les plots ont été enregistrés sous:")
    print("'",DATA_PATH,"'")

except Exception as e:
    print("L'erreur suivante est survenue :", e)


