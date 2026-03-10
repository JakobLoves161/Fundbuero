import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
from supabase import create_client, Client
import uuid

# ==============================
# 🔐 SUPABASE CONFIG
# ==============================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "clothes-images"

# ==============================
# 🧠 MODEL LOADING
# ==============================

@st.cache_resource
def load_my_model():
    return keras_load_model("keras_model.h5")

model = load_my_model()

def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels()

# ==============================
# 🎨 UI
# ==============================

st.title("👕 KI Kleidungs-Matcher")

tab1, tab2, tab3 = st.tabs([
    "🔍 Kleidung finden",
    "🚨 Verlorenes melden",
    "🖼️ Bilder aus Bucket"
])

# ==========================================================
# TAB 1 – MATCHING
# ==========================================================

with tab1:

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

    color_filter = st.selectbox(
        "Nach Farbe filtern",
        ["Alle","Blau","Rot","Schwarz","Weiß","Grün"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

        img = image.resize((224,224))
        img_array = np.array(img)
        img_array = img_array.astype(np.float32) / 127.5 - 1
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        index = np.argmax(prediction)
        predicted_class = labels[index]
        confidence = prediction[0][index]

        st.success(f"Erkannte Kategorie: {predicted_class} ({confidence*100:.2f}%)")

        query = supabase.table("clothes")\
            .select("*")\
            .eq("category", predicted_class)\
            .eq("status", "found")

        if color_filter != "Alle":
            query = query.eq("color", color_filter)

        response = query.execute()
        results = response.data

        st.subheader("🛍️ Gefundene Matches")

        if not results:
            st.warning("Keine passenden Kleidungsstücke gefunden.")

        else:
            for item in results:

                st.write(f"### {item['name']}")
                st.write(f"Farbe: {item['color']}")

                st.image(item["image_url"], width=200)

                st.markdown("---")

# ==========================================================
# TAB 2 – REPORT LOST
# ==========================================================

with tab2:

    st.subheader("Verlorenes Kleidungsstück melden")

    name = st.text_input("Name / Beschreibung")

    category = st.selectbox(
        "Kategorie",
        labels,
        key="cat_box"
    )

    color = st.selectbox(
        "Farbe",
        ["Blau","Rot","Schwarz","Weiß","Grün"],
        key="color_box"
    )

    lost_image = st.file_uploader(
        "Bild hochladen",
        type=["jpg","jpeg","png"],
        key="lost_upload"
    )

    if st.button("🚨 Als verloren melden"):

        if name and lost_image:

            file_bytes = lost_image.read()
            file_name = f"{uuid.uuid4()}.jpg"

            try:

                supabase.storage.from_(BUCKET_NAME).upload(
                    file_name,
                    file_bytes
                )

                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_name}"

                supabase.table("clothes").insert({

                    "name": name,
                    "category": category,
                    "color": color,
                    "image_url": public_url,
                    "status": "lost"

                }).execute()

                st.success("Kleidungsstück erfolgreich gemeldet!")

            except Exception as e:

                st.error(f"Fehler beim Upload: {e}")

        else:

            st.error("Bitte alle Felder ausfüllen.")

# ==========================================================
# TAB 3 – BUCKET IMAGES
# ==========================================================

with tab3:

    st.subheader("🖼️ Bilder aus Supabase Storage")

    try:

        files = supabase.storage.from_(BUCKET_NAME).list()

        if not files:

            st.info("Keine Bilder im Bucket gefunden.")

        else:

            cols = st.columns(3)

            for i,file in enumerate(files):

                file_name = file["name"]

                image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_name}"

                with cols[i % 3]:

                    st.image(image_url)
                    st.caption(file_name)

    except Exception as e:

        st.error(f"Fehler beim Laden der Bilder: {e}")
