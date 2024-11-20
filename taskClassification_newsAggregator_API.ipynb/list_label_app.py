from flask import Flask, jsonify
import joblib

app = Flask(__name__)

# Load mô hình và vectorizer đã được huấn luyện
model = joblib.load('best_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Danh sách các nhãn
labels = ['b', 'e', 'm', 't']

@app.route('/list_label', methods=['GET'])
def list_label():
    return jsonify({"labels": labels})

from flask import cli
cli.show_server_banner = lambda *_: None

if __name__ == '__main__':
    import os
    # os.environ['FLASK_ENV'] = 'development' # thêm dòng này nếu muốn chạy trong môi trường .ipynb
    app.run(host='localhost', port=2005, debug=True, use_reloader=False)