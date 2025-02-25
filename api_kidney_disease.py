from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# Charger le modèle sauvegardé
model = joblib.load("best_model_Random_Forest.pkl")  # Vérifie que ce fichier existe

# Initialiser FastAPI
app = FastAPI(
    title="API de Prédiction des Maladies Rénales",
    description="Une API permettant de prédire si un patient est atteint d'une maladie rénale ou non à partir de ses données médicales.",
    version="1.0"
)
# Monter le dossier 'static' pour le CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Définition des colonnes à supprimer pour éviter la multicolinéarité
COLUMNS_TO_DROP = ['pcv', 'bu']

# Définition des colonnes pour l'interface web
INPUT_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

# Définir la structure d'entrée attendue par l'API
class PatientData(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: str
    pc: str
    pcc: str
    ba: str
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str

def correct_incorrectly_encoded_columns(data):
    # Correction des valeurs mal encodées
    columns_to_fix = ['pcv', 'wc', 'rc', 'dm', 'cad']
    for col in columns_to_fix:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.replace("\t", "").replace("?", np.nan)
    
    # Conversion des colonnes numériques mal encodées
    cols_to_convert = ['pcv', 'wc', 'rc']
    for col in cols_to_convert:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Convertit toutes les colonnes non catégoriques (object) en float       
    for i in data.select_dtypes(exclude=['object']).columns:
        data[i] = data[i].apply(lambda x: float(x))
    
    return data
    
# Remplace les valeurs NaN par la valeur la plus fréquente (mode) dans chaque colonne.
def handle_missing_values(data):
    data_clean = correct_incorrectly_encoded_columns(data)
    mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    df_clean = pd.DataFrame(mode.fit_transform(data_clean))
    df_clean.columns = data.columns
    
    return df_clean
    
def encode_data(data):
    data_clean = handle_missing_values(data)
    encode_data = data_clean.apply(LabelEncoder().fit_transform)
    
    return encode_data  

# Fonction de prétraitement des données
def preprocess_input(data_df):
    """
    Applique le même prétraitement que celui utilisé avant l'entraînement du modèle.
    """
    try:
        df = encode_data(data_df)

        # Suppression des colonnes fortement corrélées
        df.drop(columns=COLUMNS_TO_DROP, inplace=True)

        # Normalisation des données
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_scaled = scaler.fit_transform(df)

        return df_scaled

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prétraitement : {str(e)}")
    
@app.post("/predict")
def predict(data: PatientData):
    """
    Endpoint pour la prédiction via API avec des données JSON
    """
    try:
        # Convertir les données en dictionnaire puis DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])
        
        processed_data = preprocess_input(df)
        prediction = model.predict(processed_data)
        
        return {
            "prediction": "Maladie rénale chronique" if prediction[0] == 1 else "Pas de maladie rénale chronique",
            "code": "ckd" if prediction[0] == 1 else "notckd"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Interface Web avec CSS amélioré
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="fr">
        <head>
            <title>Kidney Disease Prediction</title>
            <link rel="stylesheet" type="text/css" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h1>Prédiction de Maladie Rénale</h1>
                
                <div class="form-container">
                    <form action="/predict-form" method="post" id="predictionForm">
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="age">Âge:</label>
                                <input type="number" id="age" name="age" required step="any">
                                <div class="field-info">Âge du patient en années</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="bp">Pression artérielle:</label>
                                <input type="number" id="bp" name="bp" required step="any">
                                <div class="field-info">Pression artérielle en mm/Hg</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="sg">Gravité spécifique:</label>
                                <select id="sg" name="sg" required>
                                    <option value="1.005">1.005</option>
                                    <option value="1.010">1.010</option>
                                    <option value="1.015">1.015</option>
                                    <option value="1.020">1.020</option>
                                    <option value="1.025">1.025</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="al">Albumine:</label>
                                <select id="al" name="al" required>
                                    <option value="0">0</option>
                                    <option value="1">1</option>
                                    <option value="2">2</option>
                                    <option value="3">3</option>
                                    <option value="4">4</option>
                                    <option value="5">5</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="su">Sucre:</label>
                                <select id="su" name="su" required>
                                    <option value="0">0</option>
                                    <option value="1">1</option>
                                    <option value="2">2</option>
                                    <option value="3">3</option>
                                    <option value="4">4</option>
                                    <option value="5">5</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="rbc">Globules rouges:</label>
                                <select id="rbc" name="rbc" required>
                                    <option value="normal">Normal</option>
                                    <option value="abnormal">Anormal</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="pc">Cellules du pus:</label>
                                <select id="pc" name="pc" required>
                                    <option value="normal">Normal</option>
                                    <option value="abnormal">Anormal</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="pcc">Amas de cellules de pus:</label>
                                <select id="pcc" name="pcc" required>
                                    <option value="present">Présent</option>
                                    <option value="notpresent">Non présent</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="ba">Bactéries:</label>
                                <select id="ba" name="ba" required>
                                    <option value="present">Présent</option>
                                    <option value="notpresent">Non présent</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="bgr">Glucose sanguin:</label>
                                <input type="number" id="bgr" name="bgr" required step="any">
                                <div class="field-info">Glucose sanguin en mG/dL</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="bu">Urée sanguine:</label>
                                <input type="number" id="bu" name="bu" required step="any">
                                <div class="field-info">Urée sanguine en mG/dL</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="sc">Créatinine sérique:</label>
                                <input type="number" id="sc" name="sc" required step="any">
                                <div class="field-info">Créatinine sérique en mG/dL</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="sod">Sodium:</label>
                                <input type="number" id="sod" name="sod" required step="any">
                                <div class="field-info">Sodium en mEq/L</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="pot">Potassium:</label>
                                <input type="number" id="pot" name="pot" required step="any">
                                <div class="field-info">Potassium en mEq/L</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="hemo">Hémoglobine:</label>
                                <input type="number" id="hemo" name="hemo" required step="any">
                                <div class="field-info">Hémoglobine en gms</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="pcv">Volume cellulaire (PCV):</label>
                                <input type="number" id="pcv" name="pcv" required step="any">
                            </div>
                            
                            <div class="form-group">
                                <label for="wc">Globules blancs:</label>
                                <input type="number" id="wc" name="wc" required step="any">
                                <div class="field-info">Nombre de globules blancs (cells/cumm)</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="rc">Globules rouges:</label>
                                <input type="number" id="rc" name="rc" required step="any">
                                <div class="field-info">Nombre de globules rouges (millions/cmm)</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="htn">Hypertension:</label>
                                <select id="htn" name="htn" required>
                                    <option value="yes">Oui</option>
                                    <option value="no">Non</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="dm">Diabète:</label>
                                <select id="dm" name="dm" required>
                                    <option value="yes">Oui</option>
                                    <option value="no">Non</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="cad">Maladie coronarienne:</label>
                                <select id="cad" name="cad" required>
                                    <option value="yes">Oui</option>
                                    <option value="no">Non</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="appet">Appétit:</label>
                                <select id="appet" name="appet" required>
                                    <option value="good">Bon</option>
                                    <option value="poor">Faible</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="pe">Œdème pédieux:</label>
                                <select id="pe" name="pe" required>
                                    <option value="yes">Oui</option>
                                    <option value="no">Non</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="ane">Anémie:</label>
                                <select id="ane" name="ane" required>
                                    <option value="yes">Oui</option>
                                    <option value="no">Non</option>
                                </select>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-submit">Prédire</button>
                    </form>
                </div>
                
                <div class="file-upload">
                    <h2>Importation CSV</h2>
                    <form action="/upload-csv" method="post" enctype="multipart/form-data">
                        <input type="file" name="file" accept=".csv">
                        <button type="submit" class="btn">Prédire</button>
                    </form>
                </div>
                
                <div class="result" id="resultContainer">
                    <h3>Résultat de la prédiction</h3>
                    <p id="predictionResult"></p>
                </div>
            </div>
            
            <script>
                document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData(this);
                    const formObject = {};
                    
                    formData.forEach((value, key) => {
                        // Convert numeric strings to numbers
                        if (!isNaN(value) && value !== '') {
                            formObject[key] = parseFloat(value);
                        } else {
                            formObject[key] = value;
                        }
                    });
                    
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(formObject)
                        });
                        
                        const result = await response.json();
                        
                        const resultContainer = document.getElementById('resultContainer');
                        const predictionResult = document.getElementById('predictionResult');
                        
                        predictionResult.textContent = `Résultat: ${result.prediction}`;
                        resultContainer.style.display = 'block';
                        
                        // Scroll to result
                        resultContainer.scrollIntoView({ behavior: 'smooth' });
                        
                    } catch (error) {
                        console.error('Erreur:', error);
                        alert('Une erreur est survenue lors de la prédiction');
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/predict-form")
def predict_form(
    age: float = Form(...), bp: float = Form(...), sg: float = Form(...), al: float = Form(...),
    su: float = Form(...), rbc: str = Form(...), pc: str = Form(...), pcc: str = Form(...), ba: str = Form(...),
    bgr: float = Form(...), bu: float = Form(...), sc: float = Form(...), sod: float = Form(...),
    pot: float = Form(...), hemo: float = Form(...), pcv: float = Form(...), wc: float = Form(...), rc: float = Form(...),
    htn: str = Form(...), dm: str = Form(...), cad: str = Form(...), appet: str = Form(...),
    pe: str = Form(...), ane: str = Form(...)
):
    # Création d'un DataFrame avec les données du formulaire
    data = pd.DataFrame({
        "age": [age], "bp": [bp], "sg": [sg], "al": [al], "su": [su], 
        "rbc": [rbc], "pc": [pc], "pcc": [pcc], "ba": [ba], "bgr": [bgr], 
        "bu": [bu], "sc": [sc], "sod": [sod], "pot": [pot], "hemo": [hemo], 
        "pcv": [pcv], "wc": [wc], "rc": [rc], "htn": [htn], "dm": [dm], 
        "cad": [cad], "appet": [appet], "pe": [pe], "ane": [ane]
    })
    
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    
    return {
        "prediction": "Maladie rénale chronique" if prediction[0] == 1 else "Pas de maladie rénale chronique",
        "code": "ckd" if prediction[0] == 1 else "notckd"
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier CSV
        contents = await file.read()
        df = pd.read_csv(file.file)
        
        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_columns = [col for col in INPUT_COLUMNS if col not in df.columns]
        if missing_columns:
            return {"error": f"Colonnes manquantes dans le CSV: {', '.join(missing_columns)}"}
        
        # Prétraiter les données
        processed_data = preprocess_input(df)
        
        # Faire des prédictions
        predictions = model.predict(processed_data)
        
        # Ajouter les prédictions au DataFrame
        df['prediction'] = ["ckd" if pred == 1 else "notckd" for pred in predictions]
        
        # Convertir le DataFrame en dictionnaire pour la réponse
        results = []
        for i, row in df.iterrows():
            result = {
                "id": i,
                "prediction": "Maladie rénale chronique" if row['prediction'] == "ckd" else "Pas de maladie rénale chronique",
                "code": row['prediction']
            }
            results.append(result)
        
        return {"results": results, "total_records": len(results)}
    
    except Exception as e:
        return {"error": str(e)}

# Point d'entrée pour l'exécution du serveur
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)