import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from supabase import create_client, Client
import uuid

# ==============================
# SUPABASE CONFIG
# ==============================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "clothes-images"

# ==============================
# MODEL
# ==============================

@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5")

model = load_my_model()

def load_labels():
    labels = []
    with open("labels.txt") as f:
        for line in f:
            labels.append(line.strip().split(" ",1)[1])
    return labels

labels = load_labels()

# ==============================
# IMAGE PREPROCESS
# ==============================

def preprocess(image):

    img = image.resize((224,224))
    img_array = np.array(img)

    img_array = img_array.astype(np.float32) / 127.5 - 1
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==============================
# PREDICTION
# ==============================

def predict(image):

    processed = preprocess(image)

    prediction = model.predict(processed)

    index = np.argmax(prediction)

    label = labels[index]
    confidence = prediction[0][index]

    return label, confidence

# ==============================
# UPLOAD IMAGE
# ==============================

def upload_image(file):

    file_bytes = file.read()
    file_name = f"{uuid.uuid4()}.jpg"

    supabase.storage.from_(BUCKET).upload(
        file_name,
        file_bytes,
        {"content-type": "image/jpeg"}
    )

    url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{file_name}"

    return url

# ==============================
# UI
# ==============================

st.title("👕 KI Fundbüro für Kleidung")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Kleidung suchen",
    "🚨 Verloren melden",
    "📦 Gefunden melden",
    "🖼️ Galerie"
])

# ==============================
# SEARCH
# ==============================

with tab1:

    st.subheader("Kleidung suchen")

    uploaded = st.file_uploader("Bild hochladen", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(image)

        label, conf = predict(image)

        st.success(f"Erkannt: {label} ({conf*100:.1f}%)")

        response = supabase.table("clothes")\
            .select("*")\
            .eq("category", label)\
            .eq("status","found")\
            .execute()

        items = response.data

        if not items:

            st.warning("Keine passenden gefundenen Kleidungsstücke.")

        else:

            st.subheader("Gefundene Matches")

            for item in items:

                st.image(item["image_url"], width=200)
                st.write(item["name"])
                st.write(item["color"])
                st.divider()

# ==============================
# REPORT LOST
# ==============================

with tab2:

    st.subheader("Verlorene Kleidung melden")

    name = st.text_input("Beschreibung")

    color = st.selectbox("Farbe",
        ["Schwarz","Blau","Rot","Grün","Weiß"])

    file = st.file_uploader("Bild")

    if st.button("Verlust melden"):

        if file:

            image = Image.open(file).convert("RGB")

            label, conf = predict(image)

            url = upload_image(file)

            supabase.table("clothes").insert({

                "name": name,
                "category": label,
                "color": color,
                "image_url": url,
                "status": "lost"

            }).execute()

            st.success("Erfolgreich gemeldet!")

# ==============================
# REPORT FOUND
# ==============================

with tab3:

    st.subheader("Gefundene Kleidung melden")

    name = st.text_input("Beschreibung", key="found_name")

    color = st.selectbox("Farbe",
        ["Schwarz","Blau","Rot","Grün","Weiß"],
        key="found_color")

    file = st.file_uploader("Bild", key="found_file")

    if st.button("Fund melden"):

        if file:

            image = Image.open(file).convert("RGB")

            label, conf = predict(image)

            url = upload_image(file)

            supabase.table("clothes").insert({

                "name": name,
                "category": label,
                "color": color,
                "image_url": url,
                "status": "found"

            }).execute()

            st.success("Fund gespeichert!")

# ==============================
# GALLERY
# ==============================

with tab4:

    st.subheader("Alle Kleidungsstücke")

    response = supabase.table("clothes").select("*").execute()

    items = response.data

    if not items:

        st.info("Keine Einträge.")

    else:

        cols = st.columns(3)

        for i,item in enumerate(items):

            with cols[i%3]:

                st.image(item["image_url"])
                st.caption(item["category"])
