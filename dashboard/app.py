import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Carregar dados ---
df = pd.read_csv(r'C:\Users\55819\Downloads\projeto_pyspark\data\dados_sih\df_final.csv')
df['data'] = pd.to_datetime(df[['ano', 'mes']].rename(columns={'ano': 'year', 'mes': 'month'}).assign(day=1))

# --- Preparar dropdowns ---
anos = sorted(df['ano'].unique())
meses = sorted(df['mes'].unique())
regioes = sorted(df['regiao'].unique())
especialidades = sorted(df['especialidade'].unique())

# --- Inicializar app ---
app = dash.Dash(__name__)

# --- Layout ---
app.layout = html.Div([
    html.H1("Análise Interativa de Especialidades Hospitalares"),

    html.Div([
        html.Label("Ano"),
        dcc.Dropdown(id='filtro-ano', options=[{"label": str(a), "value": a} for a in anos], multi=True),

        html.Label("Mês"),
        dcc.Dropdown(id='filtro-mes', options=[{"label": str(m), "value": m} for m in meses], multi=True),

        html.Label("Região"),
        dcc.Dropdown(id='filtro-regiao', options=[{"label": r, "value": r} for r in regioes], multi=True),

        html.Label("Especialidade"),
        dcc.Dropdown(id='filtro-especialidade', options=[{"label": e, "value": e} for e in especialidades], value='Clínicos')
    ], style={'columnCount': 2}),

    dcc.Graph(id='grafico-linha'),
    dcc.Graph(id='grafico-distribuicao'),
    dcc.Graph(id='grafico-pizza'),

    html.H3("Tabela Interativa"),
    dash_table.DataTable(
        id='tabela',
        columns=[{"name": col, "id": col} for col in ['data', 'regiao', 'especialidade', 'qtd_mes_especialidade']],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )
])

# --- Callback para atualizar tudo ---
@app.callback(
    [
        Output('grafico-linha', 'figure'),
        Output('grafico-distribuicao', 'figure'),
        Output('grafico-pizza', 'figure'),
        Output('tabela', 'data')
    ],
    [
        Input('filtro-ano', 'value'),
        Input('filtro-mes', 'value'),
        Input('filtro-regiao', 'value'),
        Input('filtro-especialidade', 'value')
    ]
)
def atualizar_dash(ano, mes, regiao, especialidade):
    df_filtrado = df.copy()
    if ano: df_filtrado = df_filtrado[df_filtrado['ano'].isin(ano)]
    if mes: df_filtrado = df_filtrado[df_filtrado['mes'].isin(mes)]
    if regiao: df_filtrado = df_filtrado[df_filtrado['regiao'].isin(regiao)]
    if especialidade: df_filtrado = df_filtrado[df_filtrado['especialidade'] == especialidade]

    # --- Gráfico de linha com previsão ---
    df_lstm = df_filtrado[['data', 'qtd_mes_especialidade']].set_index('data').sort_index()
    if len(df_lstm) < 4:
        fig1 = go.Figure()
    else:
        scaler = MinMaxScaler()
        dados_norm = scaler.fit_transform(df_lstm)
        model = load_model(r'C:\Users\55819\Downloads\projeto_pyspark\src\meu_modelo_lstm.h5', compile=False)

        n_steps = 3
        input_seq = dados_norm[-n_steps:].reshape(1, n_steps, 1)
        preds = []
        for _ in range(12):
            p = model.predict(input_seq)[0][0]
            preds.append(p)
            input_seq = np.append(input_seq[:, 1:, :], [[[p]]], axis=1)

        preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(df_lstm.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
        df_future = pd.DataFrame({'data': future_dates, 'Predição': preds_inv}).set_index('data')
        df_plot = pd.concat([df_lstm, df_future], axis=0)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_lstm.index, y=df_lstm['qtd_mes_especialidade'], name='Real'))
        fig1.add_trace(go.Scatter(x=df_future.index, y=df_future['Predição'], name='Previsão'))
        fig1.update_layout(title='Previsão com LSTM', xaxis_title='Data', yaxis_title='Quantidade')

    # --- Gráfico de barras: Total por especialidade ---
    graf_dist = px.bar(
        df[df['regiao'].isin(regiao) if regiao else df['regiao'].unique()],
        x='especialidade', y='qtd_mes_especialidade',
        title='Distribuição de especialidades por região',
        color='regiao', barmode='group')

    # --- Gráfico de pizza ---
    pizza = df_filtrado.groupby('especialidade')['qtd_mes_especialidade'].sum().reset_index()
    graf_pizza = px.pie(pizza, names='especialidade', values='qtd_mes_especialidade',
                        title='Participação das especialidades')

    # --- Tabela ---
    tabela_data = df_filtrado[['data', 'regiao', 'especialidade', 'qtd_mes_especialidade']].to_dict('records')

    return fig1, graf_dist, graf_pizza, tabela_data

if __name__ == '__main__':
    app.run(debug=True)
