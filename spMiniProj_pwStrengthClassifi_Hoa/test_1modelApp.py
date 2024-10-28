import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('models/password_strength_model_lr.pkl')
scaler = joblib.load('models/scaler.pkl')

# Function to extract features from password
def extract_advanced_features(password):
    features = {
        'length': len(password),
        'uppercase_count': sum(1 for c in password if c.isupper()),
        'lowercase_count': sum(1 for c in password if c.islower()),
        'digit_count': sum(1 for c in password if c.isdigit()),
        'special_char_count': sum(1 for c in password if not c.isalnum()),
        'consecutive_digits_count': sum(1 for i in range(1, len(password)) if password[i].isdigit() and password[i-1].isdigit()),
        'consecutive_letters_count': sum(1 for i in range(1, len(password)) if password[i].isalpha() and password[i-1].isalpha()),
        'repeat_char_count': len(password) - len(set(password)),
        'char_variety': sum(1 for count in [sum(1 for c in password if c.isupper()), 
                                            sum(1 for c in password if c.islower()),
                                            sum(1 for c in password if c.isdigit()),
                                            sum(1 for c in password if not c.isalnum())] if count > 0)
    }
    return features

# Function to predict password strength
def predict_strength(password):
    features = extract_advanced_features(password)
    feature_array = np.array([list(features.values())])
    feature_scaled = scaler.transform(feature_array)
    strength = model.predict(feature_scaled)[0]
    return strength

# Streamlit app
st.title('Password Strength Classifier')

password = st.text_input('Enter a password:', type='password')

if st.button('Check Strength'):
    if password:
        strength = predict_strength(password)
        if strength == 0:
            st.error('Mật khẩu yếu')
        elif strength == 1:
            st.warning('Mật khẩu có độ mạnh trung bình')
        else:
            st.success('Mật khẩu mạnh')
        
        # Display feature values
        st.subheader('Các đặc trưng của mật khẩu:')
        features = extract_advanced_features(password)
        for feature, value in features.items():
            st.write(f"{feature}: {value}")
        
        # Display improvement suggestions
        st.subheader('Các gợi ý cải thiện:')
        if features['length'] < 12:
            st.write("- Tăng độ dài mật khẩu lên ít nhất 12 ký tự")
        if features['uppercase_count'] == 0:
            st.write("- Thêm ít nhất một ký tự viết hoa")
        if features['lowercase_count'] == 0:
            st.write("- Thêm ít nhất một ký tự viết thường")
        if features['digit_count'] == 0:
            st.write("- Thêm ít nhất một chữ số")
        if features['special_char_count'] == 0:
            st.write("- Thêm ít nhất một ký tự đặc biệt")
        if features['repeat_char_count'] > 2:
            st.write("- Giảm số lượng ký tự lặp lại")
    else:
        st.warning('Please enter a password')

st.markdown('---')
st.write('Ứng dụng này sử dụng một mô hình học máy để phân loại độ mạnh của mật khẩu.')

# Add usage instructions
st.sidebar.header("How to use")
st.sidebar.write("""
1. Chọn một mô hình từ danh sách.
2. Nhập mật khẩu của bạn vào ô văn bản.
3. Nhấn nút 'Kiểm tra' để xem độ mạnh của mật khẩu.
4. Xem kết quả và các chi tiết của mật khẩu.
5. Làm theo các gợi ý để cải thiện độ mạnh của mật khẩu nếu cần.
""")

# Add project information
st.sidebar.header("About the project")
st.sidebar.write("""
Dự án này sử dụng các mô hình học máy khác nhau để phân loại độ mạnh của mật khẩu dựa trên nhiều đặc trưng. 
Mục tiêu là giúp người dùng tạo ra mật khẩu mạnh hơn và an toàn hơn, đồng thời so sánh hiệu suất của các mô hình khác nhau.
""")