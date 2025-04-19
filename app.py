import joblib
import numpy as np
import streamlit as st
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
st.title("PRICE PREDICTION SYSTEM IN BANGKOK METROPOLITAN AREA")

with st.form("input_form"):
    location = st.selectbox("Location", [
    "Bang Kapi", "Bang Khen", "Bang Na", "Bang Phlat", "Bang Rak",
    "Bang Sue", "Bangkok Noi", "Chatuchak", "Chom Thong", "Din Daeng",
    "Huai Khwang", "Khan Na Yao", "Khlong San", "Khlong Toei", "Lak Si",
    "Min Buri", "Pathum Wan", "Phasi Charoen", "Phra Khanong", "Phyathai",
    "Ratchathewi", "Sathon", "Suan Luang", "Thonburi", "Wang Thonglang",
    "Watthana"])
    bedroom = st.number_input("Number of Bedroom", min_value=0, step=1)
    bathroom = st.number_input("Number of Bathroom", min_value=0, step=1)
    area_sqm = st.number_input("Room Area (sq.m)", min_value=1.0)
    distance_m = st.text_input("Distance to Station (meters)")
    station = st.selectbox("Train Station", [
    "Bang Kapi", "Ramkhamhaeng", "Huamark", "Lat Phrao 101", "Lat Pla Khao",
    "Si La Salle", "Bang Yi Khan", "Bang Phlat", "Sirindhorn", "Chong Nonsi",
    "Sam Yan", "Surasak", "lumpini", "Saint Louis", "Hua Lamphong",
    "Bang Pho", "Sala Daeng", "Bang Son", "Tao Poon", "Fai Chai",
    "Bang Khun Non", "Chatuchak Park", "Mo Chit", "Lat Phrao", "Kasetsart University",
    "Phahon Yothin", "Ratchayothin", "Saphan Khwai", "Wat Samian Nari", "Sena Nikhom",
    "Wutthakat", "Sutthisan", "Thailand Cultural Centre", "Phra Ram 9", "Ratchadaphisek",
    "Huai Khwang", "Phetchaburi", "Thong Lo", "Phawana", "Nopparat",
    "Charoen Nakhon", "Khlong Toe", "Krung Thon Buri", "Wongwian Yai", "Asok",
    "Ekkamai", "Queen Sirikit National Convention Center", "Nana", "Phra Khanong", "Phrom Phong",
    "Setthabutbamphen", "Chit Lom", "National Stadium", "On Nut", "Phloen Chit",
    "Bang Wa", "Phasi Charoen", "Ratchadamri", "Bang Chak", "Punnawithi",
    "Ari", "Udom Suk", "Sanam Pao", "Victory Monument", "Ratchaprarop",
    "Phaya Thai", "Ratchathewi", "Saphan Taksin", "Si Nut", "Talat Phlu",
    "Chok Chai 4", "Sukhumvit"])
    developer = st.selectbox("Developer", [
    "Anawat", "Grand Unity Development", "Major Development Estate", "Supalai",
    "Cube Real Property", "Assetwise", "Estate Q", "Phanalee Estate",
    "LPN Development", "ANANDA MF Asia Samyan", "Chaopraya Mahanakorn", "Nayara",
    "Pruksa Real Estate", "ANANDA Development", "Chanachai", "SC Asset Corporation",
    "Land and House", "Magnolia Quality Development Corporation", "Raimon Land",
    "Major Development", "BTS Sansiri Holding Nineteen", "SENA HANKYU 1",
    "SENA Development", "Regent Green Power", "Chewathai Interchange",
    "Richy Place (2002)", "AP (Thailand)", "Sansiri", "TEN THAI DEVELOPMENT",
    "Chewathai", "Divine Development Group", "Noble Development",
    "Plus Property Partner", "Eastern Star Real Estate", "ANANDA MF Asia Pharam 9",
    "Areeya Property", "All Inspire", "Siri TK", "A Plus Real Estate", "Built Land",
    "Praya Panich Property", "Siamese Asset", "Issara United", "Plus Property",
    "Sansiri Venture", "Property Perfect", "Big Tree Asset",
    "ANANDA MF Asia Thonglor", "Raimon Land Twenty Six", "ANANDA MF Asia Bangchak",
    "BTS Sansiri Holding Two", "39 Estate", "BTS Sansiri Holding", "The Urban Property",
    "BTS Sansiri Holding Twelve", "ANANDA MF Asia Ratchaprarop", "TCC Capital land",
    "ANANDA MF Asia Ratchathewi", "Fragrant Property", "ANANDA MF Asia Victory Monument",
    "Prinsiri", "Raimon Land Sathorn", "AP ME", "BTS Sansiri Holding Four",
    "Siri TK One", "AHJ Ekamai", "MJ One", "Nusasiri", "Major Residences",
    "ANANDA MF Asia Asoke"])
    floor = st.number_input("Your floor", min_value=1, step=1)
    total_floors = st.number_input("Floors of the building", min_value=1, step=1)
    facility = st.number_input("Facilities (0 = none, maximum 3: Pool, Fitness, Parking)", min_value=0, step=1)
    
    submitted = st.form_submit_button("PREDICTION")

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

        st.success(f"📊 PRICE PREDICTION: **{predicted_price[0]:,.2f} Bath**")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
