import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import os

# Load dữ liệu
data = pd.read_csv('processed_password_data.csv')

# Danh sách các đặc trưng
features = ['length', 'uppercase_count', 'lowercase_count', 'digit_count', 
            'special_char_count', 'repeat_char_count', 'consecutive_digits_count', 
            'consecutive_letters_count', 'char_variety']

# Khởi tạo ứng dụng Dash
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#2c3e50', 'padding': '10px'}, children=[
    html.H1("Dashboard Phân Tích Độ Mạnh Mật Khẩu", style={'textAlign': 'center', 'color': '#ecf0f1'}),
    
    # Điều chỉnh độ mạnh mật khẩu
    html.Label('Chọn độ mạnh mật khẩu:', style={'color': '#ecf0f1'}),
    dcc.RangeSlider(
        id='strength-slider',
        min=data['strength'].min(),
        max=data['strength'].max(),
        step=1,
        value=[data['strength'].min(), data['strength'].max()],
        marks={str(i): str(i) for i in range(data['strength'].min(), data['strength'].max() + 1)}
    ),

    # Dropdown chọn loại biểu đồ
    html.Label('Chọn loại biểu đồ:', style={'color': '#ecf0f1', 'margin-top': '10px'}),
    dcc.Dropdown(
        id='chart-type-dropdown',
        options=[
            {'label': 'Biểu đồ đường', 'value': 'line'},
            {'label': 'Biểu đồ phân tán', 'value': 'scatter'},
            {'label': 'Biểu đồ tròn (Pie)', 'value': 'pie'},
            {'label': 'Biểu đồ ma trận tương quan', 'value': 'heatmap'}
        ],
        value='line',
        style={'width': '50%'}
    ),

    # Dropdown chọn đặc trưng
    html.Label('Chọn đặc trưng:', style={'color': '#ecf0f1', 'margin-top': '10px'}),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in features],
        value='length',
        style={'width': '50%'}
    ),

    # Biểu đồ
    dcc.Graph(id='feature-graph', style={'padding': '20px'})
])

# Callback cập nhật biểu đồ
@app.callback(
    Output('feature-graph', 'figure'),
    [Input('strength-slider', 'value'), Input('chart-type-dropdown', 'value'), Input('feature-dropdown', 'value')]
)
def update_graph(selected_strength, chart_type, selected_feature):
    filtered_df = data[(data['strength'] >= selected_strength[0]) & (data['strength'] <= selected_strength[1])]
    
    if chart_type == 'line':
        fig = px.line(filtered_df, x='strength', y=selected_feature, color='strength', 
                      title=f'Biểu đồ Đường của {selected_feature} theo Độ Mạnh')

    elif chart_type == 'scatter':
        fig = px.scatter(filtered_df, x='strength', y=selected_feature, color='strength',
                         title=f'Biểu đồ Phân Tán của {selected_feature} theo Độ Mạnh')
    
    elif chart_type == 'pie':
        fig = px.pie(filtered_df, names='strength', title='Biểu đồ Tròn (Pie) theo Độ Mạnh')

    elif chart_type == 'heatmap':
        correlation_matrix = filtered_df[features + ['strength']].corr()
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.columns.tolist(),
            colorscale='Viridis'
        )
        fig.update_layout(title='Ma Trận Tương Quan Các Đặc Trưng và Độ Mạnh')

    fig.update_layout(plot_bgcolor='#34495e', paper_bgcolor='#2c3e50', font_color='#ecf0f1')
    return fig

# Chạy ứng dụng
server = app.server  # Cho Heroku
if __name__ == '__main__':
    app.run_server(debug=True)
