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
# 🔍 TAB 1 – SUCHEN (MIT BESCHREIBUNG)
# ==========================================================

with tab1:

    st.subheader("Kleidungsstück suchen")

    name = st.text_input("Was suchst du?")

    category = st.selectbox("Kategorie", labels)
    color = st.selectbox("Farbe", ["Alle","Blau","Rot","Schwarz","Weiß","Grün"])

    if st.button("🔍 Suchen"):

        query = supabase.table("clothes")\
            .select("*")\
            .eq("category", category)\
            .eq("status", "found")

        if color != "Alle":
            query = query.eq("color", color)

        results = query.execute().data

        st.subheader("Gefundene Matches")

        if not results:
            st.warning("Nichts gefunden.")
        else:
            for item in results:
                st.write(f"### {item['name']}")
                st.image(item["image_url"], width=200)
                st.caption(f"Farbe: {item['color']}")
                st.markdown("---")

# ==========================================================
# 📦 TAB 2 – GEFUNDEN MELDEN (NUR BILD + KI)
# ==========================================================

with tab2:

    st.subheader("Gefundenes Kleidungsstück hochladen")

    found_image = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

    if st.button("📦 Hochladen"):

        if found_image:

            image = Image.open(found_image).convert("RGB")
            st.image(image, caption="Dein Bild", use_container_width=True)

            # KI Preprocessing
            img = image.resize((224,224))
            img_array = np.array(img)
            img_array = img_array.astype(np.float32) / 127.5 - 1
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            index = np.argmax(prediction)

            predicted_class = labels[index]
            confidence = prediction[0][index]

            st.success(f"Erkannt: {predicted_class} ({confidence*100:.2f}%)")

            file_bytes = found_image.read()
            file_name = f"{uuid.uuid4()}.jpg"

            try:
                # Upload Bild
                supabase.storage.from_(BUCKET_NAME).upload(
                    file_name,
                    file_bytes
                )

                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_name}"

                # Automatischer Name
                auto_name = f"Gefunden: {predicted_class}"

                # DB speichern
                supabase.table("clothes").insert({
                    "name": auto_name,
                    "category": predicted_class,
                    "color": "Unbekannt",
                    "image_url": public_url,
                    "status": "found"
                }).execute()

                st.success("Erfolgreich gespeichert!")

            except Exception as e:
                st.error(f"Fehler: {e}")

        else:
            st.error("Bitte ein Bild hochladen.")

# ==========================================================
# 🖼️ TAB 3 – GALERIE (ALLE EINTRÄGE)
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
