# DEEP LEARNING AVEC KERAS

# But: prédiction de la catégorie de clients: détracteurs (-1), neutres (0), promoteurs (+1)
# Features: les features sont les catégories gramaticales contenus dans les commentaires (reviews) des clients
# On utilise le PCA pour construir de nouveaux features (les composants) qui sont orthogonaux entre eux

# INPUTS: data_numerique.csv (travaillé dans 'fil_rouge_data_preparation.py')
# TRAITEMENT: on utilise un modèle de regression logistique multinomiale, car il s'agit d'un problème
# de classification multiclass

# Les données ne contiennent pas des NA, autrement il aurait fallu les enlever

# Les données sont déséquilibrées
# on prend en compte ce fait à travers du paramètre 'class_weight'

# OUTPUT: un modèle de reseau neuronal que prédit de façon acceptable nos targets,
# on a réussi avoir un modèle "acceptable" mais ces résultats sont de qualité inférieure
# que les obtenus dans 'fil_rouge_prediction_deep_learning_variables_originales.py',
# autrement dit, le fait d'utiliser PCA dans ce contexte nous conduit à une perte
# de qualité au niveau de résultats

# ATTENTION: Il faut personnaliser les PATHS avant d'exécuter ce fichier

try:
    # on définit le path vers l'env virtuelle construit en local où les packages nécessaires sont installés
    # il est peut être nécessaire de personnaliser
    # imports
    import sys
    PACKAGES_PATH = 'C:\\Bib\\WPy64-3880\\notebooks\\docs\\9999_projet_fil_rouge\\pythonProject\\.venv\\Lib\\site-packages\\'
    sys.path.insert(1, PACKAGES_PATH)

    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils import class_weight
    from sklearn.decomposition import PCA

    import keras
    from tensorflow.keras import callbacks, Sequential
    from tensorflow.keras.layers import Dense, Dropout

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

    # décomposition PCA
    pca = PCA(n_components = .90)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # OHE necéessaire pour le modèle deep learning car il s'agit d'un modèle multiclass (!)
    encoder_train = OneHotEncoder()
    encoder_test = OneHotEncoder()

    Y_train = encoder_train.fit(y_train.values.reshape(-1,1))
    Y_train = Y_train.transform(y_train.values.reshape(-1,1)).toarray()

    Y_test = encoder_test.fit(y_test.values.reshape(-1,1))
    Y_test = Y_test.transform(y_test.values.reshape(-1,1)).toarray()

    # une description graphique du déséquilibre des données
    L = 1.5
    H = 0.75

    mylabels = ['Détracteurs', 'Neutres', 'Promoteurs']
    y_train_plot = pd.Series(y_train)
    title = "Data train"
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

    # sizes et dimensions
    m, n = X_train_pca.shape
    input_layer_size = int(n)  # Dimension of features
    hidden_layer_size = int(input_layer_size * 16)  # of units in hidden layer
    output_layer_size = int(len(y_train.unique()))

    # il s'agit des données qui ne sont pas équilibreés
    class_weights = class_weight.compute_class_weight('balanced', classes =np.unique(y_train), y=np.array(y_train))
    class_weights = dict(zip(np.unique(y_train), class_weights))
    # on a pu faire aussi:
    # class_weights = {}
    # for cl in np.unique(y_train):
    #     class_weights.update({cl: len(y_train) / len(y_train[y_train==cl]) / len(np.unique(y_train)) })

    # Build neural network (séquentielleà
    model = Sequential()
    model.add(keras.Input(shape=(n,), name="input_layer"))
    model.add(Dense(input_layer_size * 2, activation='relu', name="first_layer"))
    model.add(Dense(hidden_layer_size * 4, activation='relu', name="second_layer"))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_layer_size, activation='relu', name="ouput_layer"))
    model.add(Dense(n, activation='softmax'))

    # Compilation du le modèle avec la métrique appropriée (classficiation!)
    model.compile(loss='categorical_crossentropy',  # sparse_categorical_crossentropy
                  optimizer='adam',
                  metrics=['categorical_accuracy'])  # sparse_categorical_accuracy
    model.summary()

    BATCH_SIZE = 64
    EPOCHS = 20
    print("\n\n")
    print("**" * 36)
    print("BATCH_SIZE:", BATCH_SIZE, "EPOCHS: ", EPOCHS)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0.005,
                                             patience=10,
                                             mode='min',
                                             restore_best_weights=True)

    history = model.fit(X_train_pca, Y_train,
                        epochs=EPOCHS,  # 200
                        batch_size=BATCH_SIZE,
                        validation_split=0.2,
                        class_weight=class_weights,
                        callbacks=early_stopping)

    plt.figure(figsize=(12, 4))
    color = "lightgray"
    plt.rcParams.update({'axes.facecolor': color})
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte du modèle par epoch')
    plt.ylabel('perte')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

    plt.subplot(122)
    plt.rcParams.update({'axes.facecolor': color})
    plt.plot(history.history['categorical_accuracy'])  # sparse_categorical_accuracy
    plt.plot(history.history['val_categorical_accuracy'])  # val_sparse_categorical_accuracy
    plt.title('Précision du modèle par epoch')
    plt.ylabel('précision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right');

    # Prédiction à partir des données de test (attention au scaling).
    y_prob_dnn = model.predict(X_test_pca)
    y_pred_dnn = y_prob_dnn.argmax(axis=-1)

    y_pred_dnn[y_pred_dnn == 0] = -1
    y_pred_dnn[y_pred_dnn == 1] = 0
    y_pred_dnn[y_pred_dnn == 2] = 1

    # matrice de confussion
    fig = plt.figure()
    fig.patch.set_facecolor('gray')
    fig.patch.set_alpha(1.0)
    cm = confusion_matrix(y_test, y_pred_dnn)
    categ_labels = ['Detracteurs', 'Neutre', 'Promoteurs']
    # plot:
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=categ_labels, yticklabels=categ_labels)
    plt.ylabel('Prediction', fontsize=12)
    plt.xlabel('Réelle', fontsize=12)
    plt.title('Matrice de confusion', fontsize=15)
    plt.show()

    # reporting de classification
    print(classification_report(y_test, y_pred_dnn))

    print("\nLe processus et traitement se sont déroulés correctement ;-) ")

except Exception as e:
    print("L'erreur suivante est survenue :", e)