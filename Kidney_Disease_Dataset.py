# Import librairies
import numpy as np;
import pandas as pd;

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn import preprocessing
from sklearn.impute import SimpleImputer


#Loading Data and Data Pre-processing
class KIDNEY_DISEASE_DATASET():
    
    def __init__(self, path_file):
        self.path_file = path_file

    # Import Dataset
    def read_dataset(self):
        try:
            self.data = pd.read_csv(self.path_file)
            print("\nForme initiale du jeu de données:", self.data.shape)
            print("\nAfficher les premières lignes du dataset:")
            print(self.data.head())
            return self.data
        except FileNotFoundError:
            print(f"Fichier non trouvé: {self.path_file}")
            return None

    # Affichage des informations du dataset
    def data_infos(self, data,):

        # Suppression de la colonne 'id' si elle existe
        if 'id' in data.columns:
            data = data.drop('id', axis=1)

        print("\nInformation sur le dataset:")
        print(data.info())

        print("\nValeurs manquantes par colonne:")
        print(data.isnull().sum().sort_values(ascending=True))

        print("\nDescription des variables numériques du dataset:")
        print(data.describe())
        
        return data

    # Identification des colonnes numériques et catégoriques
    def check_numericals_and_categorials_columns(self, data):

        numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

        print("\nColonnes numériques:", numerical_cols)
        print("\nNombre de colonnes numériques:", len(numerical_cols))
        print("\nColonnes catégoriques:", categorical_cols)
        print("\nNombre colonnes catégoriques:", len(categorical_cols))
        

    # Vérification des valeurs uniques dans les variables catégoriques
    def check_unique_values_for_categorials(self, data):
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        print("\nValeurs uniques dans les variables catégoriques:")
        for col in categorical_cols:
            print(f"\nColonne: {col}")
            print(data[col].unique())

    # Prétraitement des données
    def correct_incorrectly_encoded_columns(self, data):

        # Correction des valeurs mal encodées
        columns_to_fix = ['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']
        for col in columns_to_fix:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip().str.replace("\t", "").replace("?", np.nan)
        
        # Conversion des colonnes numériques mal encodées
        cols_to_convert = ['pcv', 'wc', 'rc']
        for col in cols_to_convert:
            if col in self.data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convertit toutes les colonnes non catégoriques (object) en float       
        data.select_dtypes(exclude = ['object']).columns
        for i in data.select_dtypes(exclude = ['object']).columns:
            data[i] = data[i].apply(lambda x: float(x))
        
        return data
    
    # Remplace les valeurs NaN par la valeur la plus fréquente (mode) dans chaque colonne.
    def Handling_Missing_Values(self, data):
        data_clean = self.correct_incorrectly_encoded_columns(data)
        mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        df_clean = pd.DataFrame(mode.fit_transform(data_clean))
        df_clean.columns = data.columns
        
        return df_clean
    
    def encode_data(self, data):
        data_clean = self.Handling_Missing_Values(data)
        encode_data = data_clean.apply(preprocessing.LabelEncoder().fit_transform)
        
        return encode_data  
       
    # Heatmap correlation
    def correlation_columns(self, data):
        corr = data.corr()
        sns.heatmap(corr, annot = True, cmap='YlGnBu')
        plt.show()



        
path_file = 'kidney_disease.csv'        
dataset = KIDNEY_DISEASE_DATASET(path_file)

data = dataset.read_dataset()

# Afficher les informations du dataset
data = dataset.data_infos(data)

# Identifier les variables numériques et catégoriques
dataset.check_numericals_and_categorials_columns(data)


# cleanning missing values
data_clean = dataset.Handling_Missing_Values(data)


encode_data = dataset.encode_data(data)

# D’après la `matrice de corrélation`, nous observons que certaines caractéristiques présentent une `forte multicolinéarité` :
# - `pcv` et `hemo` ont une corrélation de `82 %`, indiquant une redondance d’information.
# - `sc` et `bu` ont une corrélation de `81 %` suggérant une forte interdépendance.
# - 
# Dans un modèle de Machine Learning, la présence de variables fortement corrélées peut entraîner des problèmes de multicolinéarité, ce qui peut affecter l’interprétation des coefficients et introduire du bruit dans les prédictions.

# Pour réduire la multicolinéarité, nous supprimons une caractéristique dans chaque paire fortement corrélée :
# - Suppression de `pcv`, car hemo semble mieux corrélé avec la variable cible classification (0.66).
# - Suppression de `bu`, car sc a une corrélation plus forte avec classification (0.54 contre -0.43 pour bu).

# Delete columns
# encode_data.drop('pcv', axis=1, inplace=True)
# encode_data.drop('bu', axis=1, inplace=True)

def drop_columns(data):
    
    data.drop('pcv', axis=1, inplace=True)
    data.drop('bu', axis=1, inplace=True)
    
    return data


data_final = drop_columns(encode_data)
# encode_data.to_csv("Final_pre_processing_data.csv")
