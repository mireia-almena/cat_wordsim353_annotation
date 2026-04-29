# annotation_app.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

DATASETS = {
    "Dataset 1": "word_pairs.csv",
    "Dataset 2": "word_pairs_dataset2.csv",
    "Dataset 3": "word_pairs_dataset3.csv",
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

@st.cache_resource
def get_sheet(sheet_name):
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    client = gspread.authorize(creds)
    return client.open("annotations").worksheet(sheet_name)

@st.cache_data
def load_pairs(pairs_file_path):
    return pd.read_csv(pairs_file_path)

def load_annotations(sheet):
    data = sheet.get_all_records()
    if data:
        return pd.DataFrame(data)
    return pd.DataFrame(columns=["word1", "word2", "score", "annotator"])

def save_annotation(word1, word2, score, annotator, sheet):
    sheet.append_row([word1, word2, float(score), annotator])

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Anotació de similitud entre paraules")

st.markdown("""
L'objectiu d'aquest experiment és assignar puntuacions de similitud a diversos parells de paraules per tal d'avaluar la qualitat dels models d'embeddings en català. A continuació, trobareu una llista de parells de paraules i, per a cada parell, caldrà que assigneu una puntuació numèrica de semblança en una escala de 0 a 10.

En aquesta escala, el 0 indica que les paraules no estan gens relacionades, mentre que el 10 significa que les paraules tenen una relació molt estreta. La semblança d'una paraula amb si mateixa ha de ser 10, i teniu la llibertat d'assignar puntuacions decimals, com ara un 7.5, si ho considereu oportú. Considereu els antònims com a paraules semblants. Tot i tenir significats oposats, pertanyen al mateix domini o representen característiques del mateix concepte.

Haureu d'escollir un dels tres datasets que se us proposa, i assignar les puntuacions a 100 parells de paraules. És molt important que treballeu de forma individual. No consulteu les puntuacions amb altres persones, ja que les respostes han de ser independents.

Si no coneixeu alguna de les paraules, podeu buscar-la, però si teniu qualsevol pregunta o necessiteu algun aclariment addicional, no dubteu a posar-vos en contacte amb nosaltres.

Moltes gràcies per la vostra col·laboració!
""")

selected_dataset_name = st.selectbox("Selecciona el dataset", list(DATASETS.keys()))
current_pairs_file = DATASETS[selected_dataset_name]
sheet_name = selected_dataset_name.replace(" ", "_").lower()

annotator = st.text_input("El teu nom:", key="annotator")
if not annotator:
    st.warning("Si us plau, introdueix el teu nom per començar")
    st.stop()

pairs = load_pairs(current_pairs_file)
# add these two lines:
st.write(f"Total rows loaded: {len(pairs)}")
st.write(pairs.tail(10))

try:
    sheet = get_sheet(sheet_name)
    annotations = load_annotations(sheet)
except Exception as e:
    st.error(f"Error connectant amb Google Sheets: {e}")
    st.stop()

done = set(
    zip(annotations[annotations.annotator == annotator].word1,
        annotations[annotations.annotator == annotator].word2)
)
remaining = pairs[~pairs.apply(
    lambda r: (r.word1, r.word2) in done, axis=1
)]

st.write(f"Progrés: {len(pairs) - len(remaining)} / {len(pairs)}")

if len(remaining) == 0:
    st.success("Ja has acabat! Moltes gràcies!")
    st.stop()

current = remaining.iloc[0]
st.markdown(f"## `{current.word1}`  <->  `{current.word2}`")
st.write("Quina similitud tenen aquestes dues paraules?")

score = st.slider("0 = sense relació, 10 = la mateixa paraula", 0.0, 10.0, 5.0, 0.5)

if st.button("Enviar i continuar"):
    save_annotation(current.word1, current.word2, score, annotator, sheet)
    st.cache_data.clear()
    st.rerun()
