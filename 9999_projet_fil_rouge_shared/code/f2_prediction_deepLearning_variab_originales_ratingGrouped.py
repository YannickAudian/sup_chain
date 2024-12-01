# Deep Learning (avec Keras)
# But: prédiction de la catégorie de clients: détracteurs (-1), neutres (0), promoteurs (+1)
# Features: les features sont les catégories gramaticales contenus dans les commentaires (reviews) des clients
# Pas de PCA dans ce modèle


# DEEP LEARNING WITH KERAS

# But: prédiction du 'rating'
# le rating a été groupé de la façn suivante: 1 et 2 --> 'niveau bas, 3 --> 'niveau moyen', 4 et 5 --> 'niveau haut'
# Features: les features sont les catégories grammaticales contenues dans les commentaires (reviews) des clients

# On va prédire le rating à partir du nombre de catégories grammaticales si la personne
# qui a redigé le review a donnée  a donnée un rating bas, moyen ou haut

# INPUTS: data_numerique.csv (travaillé avec 'fil_rouge_data_preparation.py')
# TRAITEMENT: on utilise u modèle séqentielle de reseau neuronal
# on applique OneHotEncoder sur le target (y)
# Les données ne contiennent des NA, autrement il aurait fallu les enlever

# Les données sont fortement déséquilibrées, c'est la raison pour laquelle on a introduit
# dans le 'fit" le paramètre "class_weight", visiblement n'était pas assez.

# OUTPUT: un modèle de reseau neuronal que prédit de façon acceptable nos targets,
# on a bien réussi à avoir ce modèle "acceptable"!

# ATTENTION: Il faut personnaliser les PATHS avant d'exécuter ce fichier

try:
    # imports
    # on définit le path vers l'env virtuelle construit en local où les packages nécessaires sont installés
    # il est peut être nécessaire de personnaliser

    import sys
    PACKAGES_PATH = 'C:\\Bib\\WPy64-3880\\notebooks\\docs\\9999_projet_fil_rouge\\pythonProject\\.venv\\Lib\\site-packages\\'
    sys.path.insert(1, PACKAGES_PATH)

    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.utils import class_weight

    import keras
    from tensorflow.keras import callbacks, Sequential
    from tensorflow.keras.layers import Dense, Dropout

    # lecture des données, personnaliser si besoin
    DATA_PATH = "C:\\Bib\\WPy64-3880\\notebooks\\docs\\9999_projet_fil_rouge\\pythonProject\\data\\"
    FILE_NAME = DATA_PATH + "data_numerique.csv"
    df = pd.read_csv(FILE_NAME,index_col=[0])
    df = df.drop(["category_nps"], axis=1)

    def grouping_rating(rating):
        """Assign a categorical description to a time."""
        if (rating == 1) | (rating == 2):
            rating_group = 0 # niveau bas
        elif (rating == 3):
            rating_group = 1 # niveau moyen
        elif (rating == 4) | (rating == 5):
            rating_group =  2 # niveau haut
        else:
            rating_group = pd.NA
        return rating_group

    # Apply la fonction sur le df
    df['rating_group'] = df['rating'].apply(grouping_rating)

    # définition target/features:
    target = df['rating_group'].copy()
    # on enleve les variables qui ont une corrélation très faible avec le target:
    feats = df.drop(['rating', 'rating_group', 'interjections_counts', 'symbols_counts', 'nb_tokens', 'plate_forme'], axis=1).copy()
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)

    # scaling la data
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()

    X_train = scaler_train.fit_transform(X_train)
    X_test = scaler_test.fit_transform(X_test)

    # juste la définition des de (le comfort!)
    X_train = pd.DataFrame(X_train, columns=feats.columns)
    X_test = pd.DataFrame(X_test, columns=feats.columns)

    # OHE nécéessaire pour le modèle deep learning car il s'agit d'un modèle multiclass (!)
    encoder_train = OneHotEncoder()
    encoder_test = OneHotEncoder()

    Y_train = encoder_train.fit(y_train.values.reshape(-1,1))
    Y_train = Y_train.transform(y_train.values.reshape(-1,1)).toarray()

    Y_test = encoder_test.fit(y_test.values.reshape(-1,1))
    Y_test = Y_test.transform(y_test.values.reshape(-1,1)).toarray()


    # plot data déséquilibrée
    L = 1.5
    H = 0.75

    mylabels = ['bas', 'moyen', 'haut']
    y_train_plot = pd.Series(y_train)
    title = "Le 'rating' groupé dans la data d'entrainement"

    plt.figure(figsize=(L,H),dpi=150)
    color = "lightgray"
    plt.figure(facecolor=color)
    y = y_train_plot.value_counts().sort_index()
    plt.pie(
        x=y,
        labels=mylabels,
        autopct='%1.2f%%',
        # Use Seaborn's color palette 'Set2'
        colors=sns.color_palette('Set2'),
        textprops={'fontsize':9}
    )
    plt.title(title)
    plt.show();

    # sizes et dimensions
    m, n = X_train.shape
    print("m: ", m, "n: ", n)
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


    # Build neural network
    model = Sequential()
    model.add(keras.Input(shape=(input_layer_size,), name="input_layer"))
    model.add(Dense(input_layer_size, activation='relu', name="first_layer"))
    model.add(Dense(hidden_layer_size * 2, activation='relu', name="second_layer"))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_layer_size * 4, activation='relu', name="ouput_layer"))
    model.add(Dense(output_layer_size, activation='softmax'))

    # Compiler le modèle avec une métrique appropriée.
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

    history = model.fit(X_train, Y_train,
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

    # Effectuer une prédiction à partir des données de test (attention au scaling).
    y_prob_dnn = model.predict(X_test)
    y_pred_dnn = y_prob_dnn.argmax(axis=-1) + 1  # c+1 car cnn commence de zéro

    # Create a confusion matrix
    fig = plt.figure()
    fig.patch.set_facecolor('gray')
    fig.patch.set_alpha(1.0)
    cm = confusion_matrix(encoder_test.inverse_transform(Y_test) + 1, y_pred_dnn)
    categ_labels = [1, 2, 3]
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=categ_labels, yticklabels=categ_labels)
    plt.ylabel('Prédiction', fontsize=12)
    plt.xlabel('Réel', fontsize=12)
    plt.title('Matrice de confusion', fontsize=15)
    plt.show()

    print(classification_report(encoder_test.inverse_transform(Y_test) + 1, y_pred_dnn))

    print("Le processus et traitement se sont déroulés correctement ;-) ")

except Exception as e:
    print("L'erreur suivante est survenue :", e)