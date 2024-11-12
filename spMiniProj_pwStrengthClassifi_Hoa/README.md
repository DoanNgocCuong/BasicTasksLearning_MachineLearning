- Python 3.12.7
- scikit-learn==1.5.0 (lúc train với lúc run app.py không nên khác phiên bản quá nhiều để tránh xung đột)


# 1. Kết quả thực nghiệm 

## 1.1 Train model với dữ liệu gốc (với các features được tạo chưa qua xử lý Outlier, Smote, ...)

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


## 1.2 Kết luận: Random Forest tốt hơn Logistic Regression và Naive Bayes.

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

========================================


## 1.3. Xử lý outliers


**1. Phương Pháp Xử Lý Outliers:**

a) **Phương pháp phát hiện và xử lý outliers bằng IQR (Interquartile Range):**
- Xác định outliers dựa trên công thức: Q1 - 1.5*IQR và Q3 + 1.5*IQR
- Áp dụng cho 9 features: length, uppercase_count, lowercase_count, digit_count, special_char_count, consecutive_digits_count, consecutive_letters_count, repeat_char_count, char_variety

Notes: Ngoài phương pháp IQR chúng ta có thể sử dụng Z-score để phát hiện và xử lý outliers.

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
- Nguyên do là phương pháp Gây mất cân bằng dữ liệu nghiêm trọng:
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

**4. Nhận xét và Đề Xuất Cải Thiện cho Outliers:**
1. Vấn đề là: Các mẫu "bất thường" có thể chứa thông tin quan trọng về mật khẩu mạnh
2. Giải pháp: 
- Thử nghiệm với phương pháp xử lý Outlier khác + kết hợp domain knowledge 
   - Thử nghiệm với ngưỡng IQR khác (2.0, 2.5)
   - Xử lý outliers theo từng class riêng biệt
   - Kết hợp domain knowledge về mật khẩu để xác định outliers
- Kết hợp các kỹ thuật xử lý mất cân bằng dữ liệu
   - Sử dụng SMOTE để tạo mẫu tổng hợp cho classes thiểu số
   - Kết hợp over-sampling và under-sampling
- Tập trung vào feature engineering và cải thiện model thêm bên cạnh việc xử lý outliers.
   - Thêm các features mới (tỷ lệ, độ phức tạp)
   - Tạo các features tương tác
   - Chuẩn hóa features theo cách khác
   - Sử dụng các thuật toán khác (XGBoost, LightGBM)
   - Tinh chỉnh hyperparameters
   - Ensemble nhiều models

================


# 2. Future Work**

1. **Xử Lý Outliers:**

   - Sử dụng các phương pháp xử lý outliers khác nhau để so sánh kết quả, như IQR, Z-score, và phương pháp khác.

   - Thử nghiệm với ngưỡng IQR khác nhau (2.0, 2.5) để xem ảnh hưởng đến hiệu suất mô hình.

   - Xử lý outliers theo từng class riêng biệt để cải thiện độ chính xác cho các lớp thiểu số.


2. **Cân Bằng Dữ Liệu:**

   - Sử dụng SMOTE để tạo mẫu tổng hợp cho các classes thiểu số và kết hợp với các phương pháp over-sampling và under-sampling khác.
   - Thử nghiệm với các kỹ thuật cân bằng dữ liệu khác nhau để đánh giá hiệu quả của chúng trên mô hình.


3. **Feature Engineering:**

   - Thêm các features mới như tỷ lệ và độ phức tạp của mật khẩu.

   - Tạo các features tương tác để cải thiện khả năng phân loại.

   - Chuẩn hóa features theo các phương pháp khác nhau để tối ưu hóa mô hình.

4. **Thử Nghiệm Mô Hình:**
   - Sử dụng các thuật toán khác như XGBoost và LightGBM để so sánh hiệu suất.

   - Tinh chỉnh hyperparameters cho các mô hình hiện tại và mới để đạt được kết quả tốt nhất.

   - Thực hiện ensemble nhiều models để cải thiện độ chính xác và độ ổn định của dự đoán.

5. **Đánh Giá và Phân Tích:**
   - Đánh giá hiệu suất của các mô hình và phương pháp xử lý khác nhau bằng các chỉ số như accuracy, precision, recall và F1-score.

   - Phân tích các mẫu "bất thường" để xác định thông tin quan trọng có thể cải thiện mô hình.

