import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from imblearn.over_sampling import RandomOverSampler

# Modèles
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


class KidneyDiseaseModelTrainer():
    def __init__(self, file_path, target_col='classification'):
        """
        Initialisation de la classe avec chargement des données et prétraitement.
        """
        self.file_path = file_path
        self.target_col = target_col
        self.models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "XGBoost": xgb.XGBClassifier()
        }
        self.results = {}

    def load_and_preprocess_data(self):
        """
        Charge et prétraite les données (équilibrage, normalisation, réduction de dimension).
        """
        # Charger les données
        data = pd.read_csv(self.file_path)
        
        data = data.drop('Unnamed: 0', axis=1)
        
        print("colonnes:", data.columns)

        # Séparer les features et la cible
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]

        # Vérifier la répartition des classes avant équilibrage
        print("Répartition avant équilibrage :", Counter(y))


        # Appliquer MinMaxScaler après équilibrage
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)

        # Séparer en ensemble d'entraînement et de test AVANT l'équilibrage
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    def train_and_evaluate(self):
        """
        Entraîne plusieurs modèles et évalue leurs performances.
        """
        for model_name, model in self.models.items():
            print(f"\n🔹 Entraînement du modèle : {model_name}")

            # Entraînement du modèle
            model.fit(self.X_train, self.y_train)

            # Prédictions
            y_pred = model.predict(self.X_test)

            # Évaluation des performances
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')  # Calcul du F1-score
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred)

            # Stocker les résultats
            self.results[model_name] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": conf_matrix,
                "classification_report": class_report
            }

            # Affichage des résultats
            print(f"✅ {model_name} - Accuracy: {accuracy:.4f} | F1-score: {f1:.4f}")
            print("\nMatrice de confusion :\n", conf_matrix)
            print("\nRapport de classification :\n", class_report)

    def compare_models(self):
        """
        Compare les modèles sur la base de l'accuracy et du f1_score.
        """
        accuracies = {model: res["accuracy"] for model, res in self.results.items()}
        f1_scores = {model: res["f1_score"] for model, res in self.results.items()}

        # Affichage des performances sous forme de graphique
        plt.figure(figsize=(12, 5))

        # Graphique des accuracy
        plt.subplot(1, 2, 1)
        sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="Blues_r")
        plt.xlabel("Modèles")
        plt.ylabel("Accuracy")
        plt.title("Comparaison des modèles - Accuracy")
        plt.xticks(rotation=30)

        # Graphique des f1-scores
        plt.subplot(1, 2, 2)
        sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette="Greens_r")
        plt.xlabel("Modèles")
        plt.ylabel("F1-score")
        plt.title("Comparaison des modèles - F1 Score")
        plt.xticks(rotation=30)

        plt.tight_layout()
        plt.show()

    def get_best_model(self):
        """
        Retourne et sauvegarde le meilleur modèle en fonction de l'accuracy et du f1_score.
        """
        best_model = max(self.results, key=lambda k: self.results[k]["f1_score"])
        best_model_instance = self.models[best_model]

        print(f"\n🏆 Le meilleur modèle est : {best_model} avec une Accuracy de {self.results[best_model]['accuracy']:.4f} et un F1-score de {self.results[best_model]['f1_score']:.4f}")

        # Sauvegarde du meilleur modèle en .pkl
        model_filename = f"best_model_{best_model}.pkl"
        joblib.dump(best_model_instance, model_filename)
        print(f"📁 Modèle sauvegardé sous : {model_filename}")

        return best_model, best_model_instance
    

model = KidneyDiseaseModelTrainer('Final_pre_processing_data.csv')

# Charger et prétraiter les données
model.load_and_preprocess_data()

# Entraîner et évaluer les modèles
model.train_and_evaluate()

# Comparer les modèles
model.compare_models()

# Sélectionner le meilleur modèle
print("🏆 Meilleur modèle :")
best_model, best_model_instance = model.get_best_model()