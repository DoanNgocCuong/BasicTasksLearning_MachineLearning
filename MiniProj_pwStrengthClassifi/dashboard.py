import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Đọc dữ liệu
data = pd.read_csv('data.csv')

# Xử lý dữ liệu
data_cleaned = data.dropna(subset=['password'])
data_cleaned.loc[:, 'password'] = data_cleaned['password'].astype(str)

# Trích xuất đặc trưng
def extract_features(password):
    return {
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

X = pd.DataFrame(data_cleaned['password'].apply(extract_features).tolist())
y = data_cleaned['strength']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình
lr_model = LogisticRegression(max_iter=1000)
nb_model = GaussianNB()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

lr_model.fit(X_train_scaled, y_train)
nb_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# Khởi tạo ứng dụng Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Phân Tích Độ Mạnh Mật Khẩu"),
    
    dcc.Tabs([
        dcc.Tab(label='Logistic Regression', children=[
            dcc.Graph(id='lr-feature-importance'),
            dcc.Graph(id='lr-confusion-matrix')
        ]),
        dcc.Tab(label='Naive Bayes', children=[
            dcc.Graph(id='nb-feature-importance'),
            dcc.Graph(id='nb-confusion-matrix')
        ]),
        dcc.Tab(label='Random Forest', children=[
            dcc.Graph(id='rf-feature-importance'),
            dcc.Graph(id='rf-confusion-matrix')
        ])
    ])
])

@app.callback(
    [Output('lr-feature-importance', 'figure'),
     Output('lr-confusion-matrix', 'figure'),
     Output('nb-feature-importance', 'figure'),
     Output('nb-confusion-matrix', 'figure'),
     Output('rf-feature-importance', 'figure'),
     Output('rf-confusion-matrix', 'figure')],
    Input('lr-feature-importance', 'id')  # Dummy input to trigger the callback
)
def update_graphs(dummy):
    # Logistic Regression
    lr_importance = pd.DataFrame({'feature': X.columns, 'importance': lr_model.coef_[0]})
    lr_importance = lr_importance.sort_values('importance', ascending=False)
    lr_fig = px.bar(lr_importance, x='feature', y='importance', title='Logistic Regression Feature Importance')
    
    lr_cm = confusion_matrix(y_test, lr_model.predict(X_test_scaled))
    lr_cm_fig = px.imshow(lr_cm, labels=dict(x="Predicted", y="Actual"), x=['Weak', 'Medium', 'Strong'], y=['Weak', 'Medium', 'Strong'],
                          title='Logistic Regression Confusion Matrix')
    
    # Naive Bayes
    nb_importance = pd.DataFrame({'feature': X.columns, 'importance': np.mean(nb_model.theta_, axis=0)})
    nb_importance = nb_importance.sort_values('importance', ascending=False)
    nb_fig = px.bar(nb_importance, x='feature', y='importance', title='Naive Bayes Feature Importance')
    
    nb_cm = confusion_matrix(y_test, nb_model.predict(X_test_scaled))
    nb_cm_fig = px.imshow(nb_cm, labels=dict(x="Predicted", y="Actual"), x=['Weak', 'Medium', 'Strong'], y=['Weak', 'Medium', 'Strong'],
                          title='Naive Bayes Confusion Matrix')
    
    # Random Forest
    rf_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
    rf_importance = rf_importance.sort_values('importance', ascending=False)
    rf_fig = px.bar(rf_importance, x='feature', y='importance', title='Random Forest Feature Importance')
    
    rf_cm = confusion_matrix(y_test, rf_model.predict(X_test_scaled))
    rf_cm_fig = px.imshow(rf_cm, labels=dict(x="Predicted", y="Actual"), x=['Weak', 'Medium', 'Strong'], y=['Weak', 'Medium', 'Strong'],
                          title='Random Forest Confusion Matrix')
    
    return lr_fig, lr_cm_fig, nb_fig, nb_cm_fig, rf_fig, rf_cm_fig

if __name__ == '__main__':
    app.run_server(debug=True)