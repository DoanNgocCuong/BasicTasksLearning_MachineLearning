import streamlit as st
import joblib
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Định nghĩa hàm để tải mô hình
def load_model(model_name):
    model_path = ""
    if model_name == 'Naive Bayes':
        model_path = 'naive_bayes_model.pkl'
    elif model_name == 'Logistic Regression':
        model_path = 'logistic_regression_model.pkl'
    elif model_name == 'Random Forest':
        model_path = 'random_forest_model.pkl'
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Mô hình {model_name} không tồn tại.")
        return None

# Định nghĩa hàm trích xuất đặc trưng từ mật khẩu
def extract_features(password):
    return np.array([[
        len(password),
        sum(1 for c in password if c.isupper()),
        sum(1 for c in password if c.islower()),
        sum(1 for c in password if c.isdigit()),
        sum(1 for c in password if not c.isalnum()),
        len(password) - len(set(password)),
        sum(1 for i in range(len(password) - 1) if password[i].isdigit() and password[i + 1].isdigit()),
        sum(1 for i in range(len(password) - 1) if password[i].isalpha() and password[i + 1].isalpha()),
        len(set(c.isupper() for c in password)) +
        len(set(c.islower() for c in password)) +
        len(set(c.isdigit() for c in password)) +
        len(set(not c.isalnum() for c in password))
    ]])

# Tải scaler đã được huấn luyện
scaler = None
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Scaler không tồn tại. Hãy đảm bảo rằng tệp 'scaler.pkl' có trong thư mục.")

# Tiêu đề ứng dụng
st.title("Kiểm tra Độ Mạnh của Mật Khẩu")

# Lựa chọn mô hình
model_name = st.selectbox("Chọn mô hình:", ["Naive Bayes", "Logistic Regression", "Random Forest"])

# Nhập mật khẩu
password = st.text_input("Nhập mật khẩu cần kiểm tra:")

# Nút kiểm tra độ mạnh
if st.button("Kiểm tra"):
    if password:
        # Trích xuất đặc trưng
        features = extract_features(password)
        
        # Kiểm tra scaler trước khi chuẩn hóa
        if scaler is not None:
            try:
                features_scaled = scaler.transform(features)
            except Exception as e:
                st.error(f"Lỗi khi chuẩn hóa dữ liệu: {e}")
                features_scaled = features  # Dùng dữ liệu chưa chuẩn hóa nếu scaler có vấn đề
            
            # Tải mô hình và dự đoán
            model = load_model(model_name)
            if model:
                try:
                    prediction = model.predict(features_scaled)[0]
                    st.write(f"Độ mạnh của mật khẩu dựa theo mô hình {model_name}: {prediction}")
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")
            else:
                st.warning("Mô hình chưa sẵn sàng.")
        else:
            st.warning("Không thể chuẩn hóa dữ liệu vì thiếu scaler.")
    else:
        st.warning("Vui lòng nhập mật khẩu.")
