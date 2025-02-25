# 🏥 Kidney Disease Prediction API & Machine Learning Models

Ce projet est une application **d'apprentissage automatique (Machine Learning)** permettant de **prédire les maladies rénales** à partir de données médicales.  
Il comprend :
- 📊 **Analyse exploratoire et prétraitement des données**
- 🤖 **Entraînement et évaluation de plusieurs modèles de Machine Learning**
- 🌍 **Une API `FastAPI` permettant d'envoyer des données patient et d'obtenir une prédiction**
- 🚀 **Déploiement possible sur un serveur**

---

## **📁 Structure du projet**

📂 Kidney_Disease_Prediction
│── kidney_disease.csv                # Initial dataset
│── Final_pre_processing_data.csv     # Processed dataset for training
│── Kidney_Disease_Dataset.py         # Data preprocessing script
│── best_model_Random_Forest.pkl      # Saved best model
│── README.md                         # Project documentation
│── setup.py                          # Installation script
│── Kidney_Disease_Prediction.py      # Model training script
│── api_kidney_disease.py             # FastAPI script for predictions
│── Kidney_Disease_Prediction.ipynb   # Jupyter notebook for analysis
│── static/style.css                  # Optional styling for the API

---

## **📦 Installation**

1️⃣ Cloner le projet

git clone https://github.com/ton-repo/kidney-disease-prediction.git
cd kidney-disease-prediction

2️⃣ Installer les dépendances

📦 Librairies utilisées

  - numpy,
  - pandas,
  - seaborn,
  - matplotlib
  - scikit-learn
  - xgboost
  - imblearn
  - fastapi,
  - uvicorn
  - joblib 

3️⃣ Prétraiter les données
`
python Kidney_Disease_Dataset.py`
`

4️⃣ Entraîner les modèles
`
python Kidney_Disease_Train.py
`
📌 Modèles entraînés

`
🌲 Decision Tree

🌳 Random Forest

🛑 SVM

📉 Logistic Regression

🔢 KNN

⚡ XGBoost
`

Le meilleur modèle est sauvegardé sous best_model.pkl.

🌍 Lancer l'API FastAPI

`
uvicorn api_kidney_disease:app --host 0.0.0.0 --port 8000 --reload
`

📌 Accès API : `http://127.0.0.1:8000`

🔥 Tester l'API avec POSTMAN ou cURL

🔹 1. Prédiction unique

📌 Endpoint : POST /predict🔹 Exemple de requête JSON :

`
{
    "age": 45, "bp": 80, "sg": 1.02, "al": 1, "su": 0,
    "rbc": "normal", "pc": "abnormal", "pcc": "notpresent",
    "ba": "notpresent", "bgr": 150, "bu": 35, "sc": 1.5,
    "sod": 140, "pot": 4.5, "hemo": 12.5, "pcv": 40, "wc": 8000,
    "rc": 4.5, "htn": "yes", "dm": "no", "cad": "no",
    "appet": "good", "pe": "no", "ane": "no"
}
`

✅ Réponse Attendue :

{
    "prediction": "ckd"
}

🔹 2. Prédictions multiples (CSV)

📌 Endpoint : POST /upload-csv
`
curl -X 'POST' 'http://127.0.0.1:8000/upload-csv' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@patients.csv'
`

🎯 Objectif

Ce projet vise à fournir un outil performant pour aider au diagnostic précoce de la maladie rénale. 🚑🔍

💡 Améliorations possibles✅ Ajout d’une interface web complète✅ Déploiement sur un serveur cloud (AWS, GCP, Azure)✅ Amélioration du modèle avec des données supplémentaires

