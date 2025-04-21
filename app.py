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
st.title("RENTAL PRICE PREDICTION SYSTEM IN BANGKOK METROPOLITAN AREA")

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
    "Ari", "Asok", "Bang Chak", "Bang Kapi", "Bang Khun Non", "Bang Pho", "Bang Phlat",
    "Bang Son", "Bang Wa", "Bang Yi Khan", "Chatuchak Park", "Charoen Nakhon", "Chit Lom",
    "Chok Chai 4", "Chong Nonsi", "Ekkamai", "Fai Chai", "Hua Lamphong", "Huai Khwang",
    "Huamark", "Kasetsart University", "Khlong Toe", "Krung Thon Buri", "Lat Phrao",
    "Lat Phrao 101", "Lat Pla Khao", "lumpini", "Mo Chit", "Nana", "National Stadium",
    "Nopparat", "On Nut", "Phetchaburi", "Phahon Yothin", "Phasi Charoen", "Phawana",
    "Phaya Thai", "Phloen Chit", "Phra Khanong", "Phra Ram 9", "Phrom Phong", "Punnawithi",
    "Queen Sirikit National Convention Center", "Ramkhamhaeng", "Ratchadamri", "Ratchadaphisek",
    "Ratchaprarop", "Ratchathewi", "Ratchayothin", "Sala Daeng", "Sam Yan", "Sanam Pao",
    "Saint Louis", "Saphan Khwai", "Saphan Taksin", "Setthabutbamphen", "Si La Salle",
    "Si Nut", "Sirindhorn", "Sukhumvit", "Surasak", "Sutthisan", "Talat Phlu", "Tao Poon",
    "Thailand Cultural Centre", "Thong Lo", "Udom Suk", "Victory Monument", "Wat Samian Nari",
    "Wongwian Yai", "Wutthakat"])
    developer = st.selectbox("Developer", [
    "39 Estate",  "A Plus Real Estate",  "AHJ Ekamai",  "All Inspire",  "ANANDA Development",  "ANANDA MF Asia Asoke",  
    "ANANDA MF Asia Bangchak",  "ANANDA MF Asia Pharam 9",  "ANANDA MF Asia Ratchaprarop",  
    "ANANDA MF Asia Ratchathewi",  "ANANDA MF Asia Samyan",  "ANANDA MF Asia Thonglor",  "ANANDA MF Asia Victory Monument",  
    "Anawat",  "AP (Thailand)",  "AP ME",  "Areeya Property",  "Assetwise",  "Big Tree Asset",  
    "Built Land",  "BTS Sansiri Holding",  "BTS Sansiri Holding Four",  "BTS Sansiri Holding Nineteen",  
    "BTS Sansiri Holding Twelve",  "BTS Sansiri Holding Two",  "Chaopraya Mahanakorn",  "Chanachai",  
    "Chewathai",  "Chewathai Interchange",  "Cube Real Property",  "Divine Development Group",  "Eastern Star Real Estate",  
    "Estate Q",  "Fragrant Property",  "Grand Unity Development",  "Issara United",  "LPN Development",  
    "Land and House",  "MJ One",  "Magnolia Quality Development Corporation",  "Major Development",  
    "Major Development Estate",  "Major Residences",  "Nayara",  "Noble Development",  "Nusasiri",  
    "Phanalee Estate",  "Plus Property",  "Plus Property Partner",  "Praya Panich Property",  "Prinsiri",  
    "Property Perfect",  "Pruksa Real Estate",  "Raimon Land",  "Raimon Land Sathorn",  "Raimon Land Twenty Six",  
    "Regent Green Power",  "SC Asset Corporation",  "SENA Development",  "SENA HANKYU 1",  
    "Sansiri",  "Sansiri Venture",  "Siamese Asset",  "Siri TK", 
    "Siri TK One",  "Supalai",  "TCC Capital land", "TEN THAI DEVELOPMENT",  "The Urban Property"])
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

        st.success(f"📊 PRICE PREDICTION: **{predicted_price[0]:,.2f} THB**")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
