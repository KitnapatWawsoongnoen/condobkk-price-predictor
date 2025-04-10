import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# โหลดโมเดล XGBoost และ Embedding
main_model = joblib.load("xgboost_model.pkl")
embedding_model = load_model("embedding_model.keras")

# โหลด encoders
le_location = joblib.load("le_location.pkl")
le_station = joblib.load("le_station.pkl")
le_developer = joblib.load("le_developer.pkl")

# ฟังก์ชันแปลงระยะทางจากตัวเลขเป็นระดับ
def distance_map_from_number(distance_m):
    try:
        distance_m = float(distance_m)
    except:
        return 1  # ถ้ากรอกผิด ให้ถือว่าเป็นระยะที่ 1

    if distance_m <= 400:
        return 3
    elif distance_m <= 1000:
        return 2
    else:
        return 1

# ฟังก์ชันแปลงชั้นที่อยู่เป็นระดับ 1-5
def map_floor_level(floor, total_floors):
    ratio = floor / total_floors
    if ratio <= 1 / 5:
        return 1
    elif ratio <= 2 / 5:
        return 2
    elif ratio <= 3 / 5:
        return 3
    elif ratio <= 4 / 5:
        return 4
    else:
        return 5

# ส่วน UI ด้วย Streamlit
st.title("🏙️ พยากรณ์ราคาคอนโดในกรุงเทพฯ")

with st.form("input_form"):
    location = st.text_input("📍 Location")
    bedroom = st.number_input("🛏️ จำนวนห้องนอน", min_value=0, step=1)
    bathroom = st.number_input("🛁 จำนวนห้องน้ำ", min_value=0, step=1)
    area_sqm = st.number_input("📐 พื้นที่ห้อง (ตร.ม.)", min_value=1.0)
    distance_m = st.text_input("🚶‍♀️ ระยะห่างจากสถานี (เมตร) (หากไม่ทราบ ให้เว้นว่าง)")
    station = st.text_input("🚆 สถานีรถไฟฟ้า")
    developer = st.text_input("🏗️ Developer")
    floor = st.number_input("🏢 ชั้นที่อยู่", min_value=1, step=1)
    total_floors = st.number_input("🏢 จำนวนชั้นทั้งหมดของตึก", min_value=1, step=1)
    facility = st.number_input("🛎️ จำนวนสิ่งอำนวยความสะดวก", min_value=0, step=1)
    
    submitted = st.form_submit_button("พยากรณ์ราคา")

if submitted:
    try:
        # Encode Categorical Features
        location_id = le_location.transform([location])[0]
        station_id = le_station.transform([station])[0]
        developer_id = le_developer.transform([developer])[0]

        # Get Embeddings
        location_emb = embedding_model.get_layer("embedding_3")(np.array([[location_id]])).numpy().reshape(-1)
        station_emb = embedding_model.get_layer("embedding_4")(np.array([[station_id]])).numpy().reshape(-1)
        developer_emb = embedding_model.get_layer("embedding_5")(np.array([[developer_id]])).numpy().reshape(-1)

        # Transform floor and distance
        level_floor = map_floor_level(floor, total_floors)
        distance_level = distance_map_from_number(distance_m)

        numerical_features = np.array([
            bedroom,
            bathroom,
            area_sqm,
            distance_level,
            level_floor,
            facility
        ])

        final_input = np.concatenate([location_emb, station_emb, developer_emb, numerical_features])
        predicted_price = main_model.predict([final_input])

        st.success(f"📊 ราคาคอนโดที่คาดการณ์: **{predicted_price[0]:,.2f} บาท**")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
