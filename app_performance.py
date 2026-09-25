"""
Application Streamlit — Prédiction Performance Etudiant Model lasso Regression
Conversion directe de l'application Gradio d'origine.

Lancement en local :  streamlit run app_performance.py
"""

import numpy as np # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
import streamlit as st # type: ignore

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predict Student Performance -Model Lasso Regression",
    page_icon="👨‍🎓",
    layout="centered",
)

DESCRIPTION = (
    "This machine learning model allows us to predict Student Performance "
    "from Hours_Studied,Previous_Scores,Extracurricular_Activities, Sleep_Hours,Sample_Question_Papers_Practiced"
)

# ----------------------------------------------------------------------
# Chargement des artefacts (mis en cache : chargés une seule fois)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    encoders = joblib.load("encoders.joblib")  # encodeurs
    uniques = joblib.load("uniques.joblib") 
    scaler = joblib.load("scaler.joblib")       # normaliseur
    lar = joblib.load("lar.joblib") 
    return encoders,uniques,scaler, lar


encoders, uniques,scaler, lar = load_artifacts()

 # type: ignore # noms des classes
# ----------------------------------------------------------------------
# Fonction de prédiction simple
# ----------------------------------------------------------------------
def Pred_func(
    Hours_Studied,
    Previous_Scores,
    Extracurricular_Activities,
    Sleep_Hours,
    Sample_Question_Papers_Practiced
):

    Extracurricular_Activities = encoders[0].transform(
        [Extracurricular_Activities]
    )[0]

    x_new = np.array([
        Hours_Studied,
        Previous_Scores,
        Extracurricular_Activities,
        Sleep_Hours,
        Sample_Question_Papers_Practiced
    ])

    x_new = x_new.reshape(1, -1)

    x_new = scaler.transform(x_new)

    y_pred = lar.predict(x_new)

    return y_pred[0]





# ----------------------------------------------------------------------
# Fonction de prédiction multiple
# ----------------------------------------------------------------------
def Pred_func_csv(file):
    # Lire le fichier csv
    df = pd.read_csv(file)
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :].values:
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4])
        predictions.append(y_pred)

    df["Selling_Price"] = predictions
    return df


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
st.title("👨‍🎓 Student Performance")

onglet1, onglet2 = st.tabs(["Simple Prediction", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Predict Student Performance with a single input")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            Previous_Scores = st.number_input("Previous Scores", value=0, step=1, format="%d")
            Extracurricular_Activities= st.selectbox("Extracurricular Activities", options=list(uniques[0])) # type: ignore
            
        with col2:
             Hours_Studied  = st.number_input("Hours Studied ", value=0, step=1, format="%d")
             Sleep_Hours = st.number_input("Sleep Hours", value=0, step=1, format="%d")
             Sample_Question_Papers_Practiced = st.number_input("Sample Question Papers Practiced", value=0, step=1, format="%d")
             
            
        soumettre = st.form_submit_button("Predict", type="primary")

    if soumettre:
        try:
            resultat = Pred_func(
            Hours_Studied,
            Previous_Scores,
            Extracurricular_Activities,
            Sleep_Hours,
            Sample_Question_Papers_Practiced
        )

            st.success(
            f"**Performance Index prédit : {resultat:.2f}**"
        )

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Predict Student Performance with multiple inputs")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "Hours_Studied,Previous_Scores,Extracurricular_Activities, Sleep_Hours,Sample_Question_Papers_Practiced"
    )

    fichier = st.file_uploader("Upload a csv file", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Download a csv file",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
