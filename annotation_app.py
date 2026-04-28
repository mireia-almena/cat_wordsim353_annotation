# annotation_app.py
import streamlit as st
import pandas as pd
import os

DATASETS = {
    "Dataset 1": "word_pairs.csv",
    "Dataset 2": "word_pairs_dataset2.csv",
    "Dataset 3": "word_pairs_dataset3.csv",
}

@st.cache_data
def load_pairs(pairs_file_path):
    if not os.path.exists(pairs_file_path):
        st.error(f"Critical Error: File '{pairs_file_path}' not found. Please ensure it exists in the app directory.")
        st.info("The file should contain columns: 'word1' and 'word2'")
        st.stop()
    return pd.read_csv(pairs_file_path)

def load_annotations(annotations_file_path):
    if os.path.exists(annotations_file_path):
        return pd.read_csv(annotations_file_path)
    return pd.DataFrame(columns=["word1", "word2", "score", "annotator"])

def save_annotation(word1, word2, score, annotator, annotations_file_path):
    new_row = pd.DataFrame([{
        "word1": word1, "word2": word2,
        "score": score, "annotator": annotator
    }])
    # Use append mode to be more efficient and avoid rewriting the entire file
    file_exists = os.path.isfile(annotations_file_path)
    new_row.to_csv(annotations_file_path, mode='a', index=False, header=not file_exists)

st.title("Word Similarity Annotation")

st.markdown("""
L'objectiu d'aquest experiment és assignar puntuacions de similitud a diversos parells de paraules per tal d'avaluar la qualitat dels models d'embeddings en català. A continuació, trobareu una llista de parells de paraules i, per a cada parell, caldrà que assigneu una puntuació numèrica de semblança en una escala de 0 a 10. 

En aquesta escala, el 0 indica que les paraules no estan gens relacionades, mentre que el 10 significa que les paraules tenen una relació molt estreta. La semblança d’una paraula amb si mateixa ha de ser 10, i teniu la llibertat d'assignar puntuacions decimals, com ara un 7.5, si ho considereu oportú. Considereu els antònims com a paraules semblants. Tot i tenir significats oposats, pertanyen al mateix domini o representen característiques del mateix concepte.

Haureu d'escollir un dels tres datasets que se us proposa, i assignar les puntuacions a 100 parells de paraules. És molt important que treballeu de forma individual. No consulteu les puntuacions amb altres persones, ja que les respostes han de ser independents.

Si teniu qualsevol pregunta o necessiteu algun aclariment addicional, no dubteu a posar-vos en contacte amb nosaltres. 

Moltes gràcies per la vostra col·laboració! 
""")

selected_dataset_name = st.selectbox("Select Dataset", list(DATASETS.keys()))
current_pairs_file = DATASETS[selected_dataset_name]
current_annotations_file = f"annotations_{selected_dataset_name.replace(' ', '_').lower()}.csv"

annotator = st.text_input("El teu nom:", key="annotator")
if not annotator:
    st.warning("Si us plau, introdueix el teu nom per començar")
    st.stop()

pairs = load_pairs(current_pairs_file)
annotations = load_annotations(current_annotations_file)

done = set(
    zip(annotations[annotations.annotator == annotator].word1,
        annotations[annotations.annotator == annotator].word2)
)
remaining = pairs[~pairs.apply(
    lambda r: (r.word1, r.word2) in done, axis=1
)]

st.write(f"Progress: {len(pairs) - len(remaining)} / {len(pairs)}")

if len(remaining) == 0:
    st.success("Ja has acabat! Moltes gràcies!")
    st.stop()

current = remaining.iloc[0]
st.markdown(f"## `{current.word1}`  <->  `{current.word2}`")
st.write("Quina similitud tenen aquestes dues paraules?")

score = st.slider("0 = sense relació, 10 = la mateixa paraula", 0.0, 10.0, 5.0, 0.5)

if st.button("Enviar i continuar"):
    save_annotation(current.word1, current.word2, score, annotator, current_annotations_file)
    st.rerun()
