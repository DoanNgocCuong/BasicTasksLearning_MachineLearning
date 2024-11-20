from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load mô hình và vectorizer đã được huấn luyện
model = joblib.load('best_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.route('/classify', methods=['POST'])
def classify():
    # Lấy dữ liệu từ request
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Chuyển đổi văn bản thành vector
    text_vector = vectorizer.transform([text])

    # Dự đoán nhãn và xác suất
    predicted_label = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Lấy xác suất cao nhất
    max_prob = np.max(probabilities)

    # Chuẩn bị response
    response = {
        "text": text,
        "predicted_label": predicted_label,
        "prob": float(max_prob)
    }

    return jsonify(response)

if __name__ == '__main__':
    import os   
    # os.environ['FLASK_ENV'] = 'development'   # Thêm dòng này nếu muốn chạy trong môi trường .ipynb
    app.run(host='localhost', port=2005, debug=True, use_reloader=False)