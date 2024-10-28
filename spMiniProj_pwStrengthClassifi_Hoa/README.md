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