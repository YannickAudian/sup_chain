# REGRESSION LOGISTIQUE MULTINOMIALE

# But: prédiction de la catégorie de clients: détracteurs (-1), neutres (0), promoteurs (+1)

# Features: les features sont les catégories gramaticales contenus dans les commentaires (reviews) des clients
# dû à la présence de multicollinéarite on utilise le PCA pour construir des nouveaux features (les composants)
# qui sont orthogonaux entre eux

# On calcule les poids des composants dans la régression et le poids des variables originales dans
# la construction de ces composants

# INPUTS: data_numerique.csv (travaillé dans 'fil_rouge_data_preparation.py')
# TRAITEMENT: on utilise un modèle de regression logistique multinomiale, car il s'agit d'un problème
# de classification multiclass

# Les données ne contiennent des NA, autrement il aurait fallu les enlever

# Les données sont déséquilibrées

# OUTPUT: un modèle de reseau neuronal que prédit de façon acceptable nos targets,
# on a réussi avoir un modèle "acceptable"

# On calcule les poids des composants dans la régression et le poids des variables originales dans
# la construction de ces composants
# ATTENTION: Il faut personnaliser les PATHS avant d'exécuter ce fichier

# on définit le path vers l'env virtuelle construit en local où les packages nécessaires sont installés
# il est peut être nécessaire de personnaliser

# REGRESSION LOGISTIQUE MULTINOMIALE

# But: prédiction de la catégorie de clients: détracteurs (-1), neutres (0), promoteurs (+1)

# Features: les features sont les catégories gramaticales contenus dans les commentaires (reviews) des clients
# dû à la présence de multicollinéarite on utilise le PCA pour construir des nouveaux features (les composants)
# qui sont orthogonaux entre eux

# On calcule les poids des composants dans la régression et le poids des variables originales dans
# la construction de ces composants

# INPUTS: data_numerique.csv (travaillé dans 'fil_rouge_data_preparation.py')
# TRAITEMENT: on utilise un modèle de regression logistique multinomiale, car il s'agit d'un problème
# de classification multiclass

# Les données ne contiennent des NA, autrement il aurait fallu les enlever

# Les données sont déséquilibrées

# OUTPUT: un modèle de reseau neuronal que prédit de façon acceptable nos targets,
# on a réussi avoir un modèle "acceptable"

# On calcule les poids des composants dans la régression et le poids des variables originales dans
# la construction de ces composants
# ATTENTION: Il faut personnaliser les PATHS avant d'exécuter ce fichier

# on définit le path vers l'env virtuelle construit en local où les packages nécessaires sont installés
# il est peut être nécessaire de personnaliser

try:
    # imports
    import sys
    PACKAGES_PATH = 'C:\\Bib\\WPy64-3880\\notebooks\\docs\\9999_projet_fil_rouge\\pythonProject\\.venv\\Lib\\site-packages\\'
    sys.path.insert(1, PACKAGES_PATH)

    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.inspection import permutation_importance

    # lecture des données, personnaliser si besoin
    DATA_PATH = "C:\\Bib\\WPy64-3880\\notebooks\\docs\\9999_projet_fil_rouge\\pythonProject\\data\\"
    FILE_NAME = DATA_PATH + "data_numerique.csv"
    df = pd.read_csv(FILE_NAME,index_col=[0])
    df = df.drop(["category_nps"], axis=1)

    # définition target/features:
    target = df['category_numeric'].copy()
    # on enleve les variables qui ont une corrélation très faible avec le target:
    feats = df.drop(['category_numeric', 'interjections_counts', 'symbols_counts', 'nb_tokens', 'plate_forme'], axis=1).copy()
    # split trian/test
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)

    # scaling la data
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()

    X_train = scaler_train.fit_transform(X_train)
    X_test = scaler_test.fit_transform(X_test)

    # juste la définition des de (le comfort!)
    X_train = pd.DataFrame(X_train, columns=feats.columns)
    X_test = pd.DataFrame(X_test, columns=feats.columns)

    # une description graphique du déséquilibre des données
    L = 1.5
    H = 0.75

    mylabels = ['Détracteurs', 'Neutres', 'Promoteurs']
    y_train_plot = pd.Series(y_train)
    title = "Les types de clients dans la data d'entrainement"
    plt.figure(figsize=(L, H), dpi=150)
    color = "lightgray"
    plt.figure(facecolor=color)
    y = y_train_plot.value_counts().sort_index()
    plt.pie(
        x=y,
        labels=mylabels,
        autopct='%1.2f%%',
        # Use Seaborn's color palette 'Set2'
        colors=sns.color_palette('Set2'),
        textprops={'fontsize': 9}
    )
    plt.title(title)
    plt.show();



    # PCA
    pca = PCA(n_components = .90)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # plot: contribution des composants à la variance expliquée
    X_train.columns
    var_ratio = []
    for num in range(0, 4):
        pca = PCA(n_components=num)
        pca.fit(X_train)
        var_ratio.append(np.sum(pca.explained_variance_ratio_))

    plt.figure(figsize=(4,2),dpi=150)
    color = "lightgray"
    plt.figure(facecolor=color)
    plt.grid()
    plt.plot([0,1,2,3],var_ratio,marker='o')
    plt.xlabel('n_components')
    plt.ylabel('Explained variance ratio')
    plt.title('n_components vs. Explained Variance Ratio')
    plt.show()


    # plot: contribution des variables originals aux composants
    df_pca_components = pd.DataFrame(pca.components_, columns=X_train.columns, index=['PCA0', 'PCA1', 'PCA2'])
    legend_names = df_pca_components.columns.to_list()
    plt.figure(figsize=(12,4),dpi=150)
    color = "lightgray"
    plt.figure(facecolor=color)
    df_pca_components.plot(kind='bar', stacked=True, legend=True);
    plt.xlabel("composants")
    plt.ylabel("")
    plt.legend(legend_names)
    plt.title("Contribution des 'features' sur les PCA")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
    plt.tight_layout()
    plt.show();

    # Définition du modèle de regréssion logistique multinomiale (avec cross validation)
    model_lg = LogisticRegressionCV(cv=10, random_state=0).fit(X_train_pca, y_train)

    # accuracy su modèle
    print("\nAccuracy su modèle")
    print("score sur le training data: ", model_lg.score(X_train_pca, y_train))
    predict_lg = model_lg.predict(X_test_pca)
    print("score sur le testing data: ", accuracy_score(y_test, predict_lg))

    # calcul et plot de la matrice de confusion
    fig = plt.figure()
    fig.patch.set_facecolor('gray')
    fig.patch.set_alpha(1.0)
    cm = confusion_matrix(y_test, predict_lg)
    categ_labels = ['Detracteurs', 'Neutre', 'Promoteurs']
    sns.heatmap(cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=categ_labels, yticklabels=categ_labels)
    plt.ylabel('Prediction',fontsize=12)
    plt.xlabel('Actual',fontsize=12)
    plt.title('Matrice de confusion',fontsize=15)
    plt.show()

    # classification report
    print("\nClassification report")
    print(classification_report(y_test, predict_lg))

    # feature importance -- PCA
    coefficients = model_lg.coef_[0]
    df_coefficients = pd.DataFrame(coefficients, columns=['values'], index=['PCA0', 'PCA1', 'PCA2'])

    color = "lightgray"
    fig = plt.figure(figsize = (10, 4))
    plt.figure(facecolor=color)
    # creating the bar plot
    x = df_coefficients.index
    y = coefficients
    plt.bar(x,y)
    plt.xlabel("régresseurs")
    plt.ylabel("values")
    plt.title("Coefficients de la régression logistique")
    plt.show()

    # proportion de la variance expliquée du modèle par les composants
    columns = ['PCA1', 'PCA2', 'PCA3']
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=columns)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=columns)

    fig = plt.figure()
    fig.patch.set_facecolor('gray')
    fig.patch.set_alpha(1.0)
    plt.plot(pca.explained_variance_ratio_)
    plt.ylabel('Proportion',fontsize=12)
    plt.xlabel('Composantes',fontsize=12)
    plt.xticks([0,1,2,3],['PC1', 'PC2', 'PC3', 'PC4'])
    plt.title('Composantes principales: proportion de la variance expliquée',fontsize=15)
    plt.show()

    # analyse de l'interprétabilité du modèle
    model = LogisticRegressionCV(cv=10, random_state=0).fit(X_train_pca, y_train)
    r = permutation_importance(model, X_train_pca, y_train,
                               n_repeats=30,
                               random_state=0)

    print("Interprétabilité du modèle:")
    print("importances mean:", r.importances_mean)
    print("importances std:", r.importances_std)

    print("\nLe processus et traitement se sont déroulés correctement ;-) ")

except Exception as e:
    print("L'erreur suivante est survenue :", e)