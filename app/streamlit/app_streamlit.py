import streamlit as st
from src.diabetes.predict import predict_diabetes

st.set_page_config(page_title="당뇨병 예측 시스템", page_icon="🩺")

st.title("🩺 당뇨병 예측 시스템")
st.write("아래 정보를 입력하고 예측 결과를 확인해보세요.")

# 사용자 입력 받기
gender = st.radio("성별", ["여성", "남성"])
age = st.number_input("나이", min_value=1, max_value=120, value=50)
hypertension = st.radio("고혈압 유무", ["없음", "있음"])
heart_disease = st.radio("심장병 유무", ["없음", "있음"])
smoking_level = st.selectbox("흡연 척도", [
    "비흡연 (0)", "5년 이상 금연 (1)", "흡연 중 (2)", "5년 이내 금연 (3)"
])
bmi = st.number_input("BMI 지수", min_value=10.0, max_value=50.0, value=25.0)
hba1c_level = st.number_input("HbA1c 수치", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input("혈당 수치", min_value=50.0, max_value=300.0, value=100.0)

# 값 변환
user_input = {
    'gender': 1 if gender == "남성" else 0,
    'age': age,
    'hypertension': 1 if hypertension == "있음" else 0,
    'heart_disease': 1 if heart_disease == "있음" else 0,
    'smoking_history': int(smoking_level.split("(")[-1][0]),  # 0~3
    'bmi': bmi,
    'HbA1c_level': hba1c_level,
    'blood_glucose_level': blood_glucose_level
}

if st.button("📊 예측 실행"):
    probability = predict_diabetes(user_input)
    st.subheader(f"🧾 예측된 당뇨병 발병 확률: **{probability * 100:.2f}%**")

    if probability >= 0.7:
        st.error("⚠️ 고위험군입니다. 가까운 병원 방문을 권장합니다.")
    elif probability >= 0.4:
        st.warning("🟡 주의가 필요합니다. 건강 습관을 개선해 주세요.")
    else:
        st.success("🟢 현재는 낮은 위험도입니다.")


# 실행코드
# streamlit run app/diabetes_app/streamlit/app_streamlit.py