- Python 3.12.7
- scikit-learn==1.5.0 (lúc train với lúc run app.py không nên khác phiên bản quá nhiều để tránh xung đột)


#### Kết quả 

- Logistic Regression: `model = LogisticRegression(multi_class='ovr', max_iter=1000)`
```
Độ chính xác: 0.7588331043545786
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         1
           0       0.29      0.02      0.03     17999
           1       0.78      0.96      0.86     99267
           2       0.60      0.38      0.47     16661

    accuracy                           0.76    133928
   macro avg       0.42      0.34      0.34    133928
weighted avg       0.69      0.76      0.70    133928
```

- Naive Bayes: `nb_model = GaussianNB()`
```
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         1
           0       0.52      0.48      0.50     17999
           1       0.86      0.86      0.86     99267
           2       0.55      0.59      0.57     16661

    accuracy                           0.78    133928
   macro avg       0.48      0.48      0.48    133928
weighted avg       0.77      0.78      0.77    133928

```

- Random Forest: `rf_model = RandomForestClassifier(n_estimators=100, random_state=42)`
```
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         1
           0       0.61      0.61      0.61     17999
           1       0.88      0.89      0.88     99267
           2       0.60      0.55      0.57     16661

    accuracy                           0.81    133928
   macro avg       0.52      0.51      0.52    133928
weighted avg       0.81      0.81      0.81    133928
```


Random Forest tốt hơn Logistic Regression và Naive Bayes.
============
```
1. So sánh tổng quát:
Random Forest có hiệu suất tốt nhất với độ chính xác (accuracy) 0.81.
Tiếp theo là Naive Bayes với độ chính xác 0.78.
Cuối cùng là Logistic Regression với độ chính xác 0.7588.

2. Phân tích chi tiết:
a) Random Forest:
Hiệu suất tốt nhất trên tất cả các lớp.
Cân bằng tốt giữa precision và recall cho mỗi lớp.
F1-score cao nhất cho lớp 1 (0.88) và lớp 0 (0.61).
b) Naive Bayes:
Hiệu suất tốt thứ hai.
Cân bằng khá tốt giữa precision và recall.
Đặc biệt tốt trong việc phân loại lớp 1 (F1-score 0.86).
c) Logistic Regression:
Hiệu suất thấp nhất trong ba mô hình.
Gặp khó khăn trong việc phân loại lớp 0 (F1-score chỉ 0.03).
Có xu hướng phân loại nhiều mẫu vào lớp 1 (recall 0.96 cho lớp 1).
```




===================

# Xử lý outliers

**REPORT: PHÂN TÍCH XỬ LÝ OUTLIERS TRONG DỮ LIỆU MẬT KHẨU**

**1. Phương Pháp Xử Lý Outliers:**

a) **Phương pháp IQR (Interquartile Range):**
- Xác định outliers dựa trên công thức: Q1 - 1.5*IQR và Q3 + 1.5*IQR
- Áp dụng cho 9 features: length, uppercase_count, lowercase_count, digit_count, special_char_count, consecutive_digits_count, consecutive_letters_count, repeat_char_count, char_variety

b) **Hai cách xử lý được thử nghiệm:**
1. **Clip Method:**
   - Giữ nguyên số lượng dòng dữ liệu
   - Điều chỉnh giá trị outliers về giới hạn trên/dưới
   - Dữ liệu ban đầu: 669,640 dòng → Sau xử lý: 669,640 dòng

2. **Remove Method:**
   - Loại bỏ hoàn toàn các dòng chứa outliers
   - Dữ liệu ban đầu: 669,640 dòng → Sau xử lý: 524,212 dòng
   - Tỷ lệ loại bỏ theo features:
     + length: 9.33%
     + uppercase_count: 4.19%
     + lowercase_count: 4.27%
     + digit_count: 2.57%
     + special_char_count: 0.93%
     + consecutive_digits_count: 0%
     + consecutive_letters_count: 0%
     + repeat_char_count: 0.29%
     + char_variety: 0.13%

**2. Kết Quả và Đánh Giá:**

a) **Phương pháp Clip:**
```
Accuracy: 0.8084 (≈ 80.84%)
              precision    recall  f1-score
0                 0.61      0.61      0.61
1                 0.88      0.89      0.88
2                 0.60      0.55      0.57
```
- Không cải thiện đáng kể so với dữ liệu gốc
- Giữ được toàn bộ dữ liệu
- Phân phối classes không thay đổi

b) **Phương pháp Remove:**
```
Accuracy: 0.7585 (≈ 75.85%)
              precision    recall  f1-score
0                 0.61      0.67      0.63
1                 0.88      0.82      0.85
2                 0.06      0.10      0.07
```
- Hiệu suất giảm so với dữ liệu gốc
- Gây mất cân bằng dữ liệu nghiêm trọng:
  + Class 1: 416,042 mẫu (79.4%)
  + Class 0: 78,644 mẫu (15.0%)
  + Class 2: 29,526 mẫu (5.6%)

**3. Nhận Xét:**

1. **Clip Method:**
   - Ưu điểm:
     + Không mất mẫu dữ liệu
     + Giữ được phân phối classes
   - Nhược điểm:
     + Không cải thiện được hiệu suất model
     + Có thể làm mất đi thông tin quan trọng về các mẫu đặc biệt

2. **Remove Method:**
   - Ưu điểm:
     + Loại bỏ được các giá trị bất thường
     + Dữ liệu "sạch" hơn
   - Nhược điểm:
     + Làm mất cân bằng dữ liệu nghiêm trọng
     + Giảm hiệu suất model, đặc biệt với class 2
     + Mất nhiều mẫu dữ liệu có thể quan trọng

**4. Đề Xuất Cải Thiện:**

1. **Cân Bằng Dữ liệu:**
   - Sử dụng SMOTE để tạo mẫu tổng hợp cho classes thiểu số
   - Kết hợp over-sampling và under-sampling

2. **Điều Chỉnh Phương Pháp:**
   - Thử nghiệm với ngưỡng IQR khác (2.0, 2.5)
   - Xử lý outliers theo từng class riêng biệt
   - Kết hợp domain knowledge về mật khẩu để xác định outliers

3. **Feature Engineering:**
   - Thêm các features mới (tỷ lệ, độ phức tạp)
   - Tạo các features tương tác
   - Chuẩn hóa features theo cách khác

4. **Thử Nghiệm Mô Hình:**
   - Sử dụng các thuật toán khác (XGBoost, LightGBM)
   - Tinh chỉnh hyperparameters
   - Ensemble nhiều models

**5. Kết Luận:**
- Với đặc thù của bài toán phân loại độ mạnh mật khẩu, việc xử lý outliers cần cân nhắc kỹ
- Các mẫu "bất thường" có thể chứa thông tin quan trọng về mật khẩu mạnh
- Nên kết hợp domain knowledge và các kỹ thuật xử lý mất cân bằng dữ liệu
- Tập trung vào feature engineering và cải thiện model thay vì xử lý outliers đơn thuần
