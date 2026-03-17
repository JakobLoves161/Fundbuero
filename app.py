import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
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
# 🧠 MODEL
# ==============================

@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5")

model = load_my_model()

def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels()

# ==============================
# 🎨 UI
# ==============================

st.title("👕 Lost & Found KI App")

tab1, tab2, tab3 = st.tabs([
    "🔍 Suchen",
    "📦 Gefunden melden",
    "🖼️ Galerie"
])

# ==========================================================
# 🔍 TAB 1 – SUCHEN (KLASSIFIZIEREN)
# ==========================================================

with tab1:

    st.subheader("Kleidungsstück suchen")

    uploaded_file = st.file_uploader("Bild zum Suchen hochladen", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Dein Bild", use_container_width=True)

        # Preprocessing
        img = image.resize((224,224))
        img_array = np.array(img)
        img_array = img_array.astype(np.float32) / 127.5 - 1
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        index = np.argmax(prediction)

        predicted_class = labels[index]
        confidence = prediction[0][index]

        st.success(f"Erkannt: {predicted_class} ({confidence*100:.2f}%)")

        # DB Query → NUR gefundene Sachen
        response = supabase.table("clothes")\
            .select("*")\
            .eq("category", predicted_class)\
            .eq("status", "found")\
            .execute()

        results = response.data

        st.subheader("🔎 Gefundene Matches")

        if not results:
            st.warning("Keine passenden Items gefunden.")

        else:
            for item in results:
                st.write(f"### {item['name']}")
                st.write(f"Kategorie: {item['category']}")
                st.write(f"Farbe: {item['color']}")
                st.image(item["image_url"], width=200)
                st.markdown("---")


# ==========================================================
# 📦 TAB 2 – GEFUNDEN MELDEN (UPLOAD)
# ==========================================================

with tab2:

    st.subheader("Gefundenes Kleidungsstück melden")

    name = st.text_input("Beschreibung")

    category = st.selectbox("Kategorie", labels)
    color = st.selectbox("Farbe", ["Blau","Rot","Schwarz","Weiß","Grün"])

    found_image = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

    if st.button("📦 Als gefunden speichern"):

        if name and found_image:

            file_bytes = found_image.read()
            file_name = f"{uuid.uuid4()}.jpg"

            try:
                # Upload
                supabase.storage.from_(BUCKET_NAME).upload(
                    file_name,
                    file_bytes
                )

                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_name}"

                # DB Eintrag
                supabase.table("clothes").insert({
                    "name": name,
                    "category": category,
                    "color": color,
                    "image_url": public_url,
                    "status": "found"
                }).execute()

                st.success("Erfolgreich gespeichert!")

            except Exception as e:
                st.error(f"Fehler: {e}")

        else:
            st.error("Bitte alles ausfüllen.")


# ==========================================================
# 🖼️ TAB 3 – GALERIE (ALLE BILDER)
# ==========================================================

with tab3:

    st.subheader("Alle hochgeladenen Kleidungsstücke")

    try:
        response = supabase.table("clothes").select("*").execute()
        items = response.data

        if not items:
            st.info("Noch keine Einträge vorhanden.")

        else:
            cols = st.columns(3)

            for i, item in enumerate(items):

                with cols[i % 3]:
                    st.image(item["image_url"])
                    st.caption(f"{item['name']} ({item['status']})")

    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
