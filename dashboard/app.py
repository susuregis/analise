import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# --- Leitura do CSV com previsão ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_path = os.path.join(BASE_DIR, '..', 'data', 'dados_sih', 'df_final.csv')
df = pd.read_csv(df_path)

# --- Gerar previsão para 2026 ---
df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str).str.zfill(2))
df_grouped = df.groupby(['data', 'especialidade'])['qtd_mes_especialidade'].sum().reset_index()

especialidades = df_grouped['especialidade'].unique()
forecast_list = []

for esp in especialidades:
    df_esp = df_grouped[df_grouped['especialidade'] == esp].set_index('data').asfreq('MS')
    df_esp['qtd_mes_especialidade'].interpolate(inplace=True)

    model = SARIMAX(df_esp['qtd_mes_especialidade'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=12)
    forecast_df = forecast.predicted_mean.reset_index()
    forecast_df.columns = ['data', 'qtd_mes_especialidade']
    forecast_df['especialidade'] = esp

    # --- Garantir que todos os meses de 2026 estejam no forecast_df ---
    idx_2026 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
    forecast_df = forecast_df.set_index('data').reindex(idx_2026).reset_index()
    forecast_df.rename(columns={'index': 'data'}, inplace=True)
    forecast_df['especialidade'] = esp

    # Preencher NaNs se existirem (por segurança)
    forecast_df['qtd_mes_especialidade'].interpolate(inplace=True)
    
    forecast_list.append(forecast_df)

df_forecast = pd.concat(forecast_list, ignore_index=True)
df_forecast['ano'] = df_forecast['data'].dt.year
df_forecast['mes'] = df_forecast['data'].dt.month
df_forecast['regiao'] = 'Previsão'

# Concatenar previsões com dados reais
df = pd.concat([
    df[['ano', 'mes', 'especialidade', 'qtd_mes_especialidade', 'regiao']],
    df_forecast[['ano', 'mes', 'especialidade', 'qtd_mes_especialidade', 'regiao']]
], ignore_index=True)

# --- Conversão de tipos ---
df['ano'] = df['ano'].astype(str)
df['mes'] = df['mes'].astype(str)

# --- Inicialização do app ---
app = Dash(__name__)
app.title = 'Dashboard de Especialidades Hospitalares'

# --- Layout ---
app.layout = html.Div([
    html.H1('Análise Interativa de Especialidades Hospitalares'),

    html.Div([
        html.Div([
            html.Label('Ano'),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in sorted(df['ano'].unique())],
                id='filtro_ano',
                placeholder='Select...'
            )
        ], style={'width': '24%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Mês'),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in sorted(df['mes'].unique())],
                id='filtro_mes',
                placeholder='Select...'
            )
        ], style={'width': '24%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Região'),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in df['regiao'].unique()],
                id='filtro_regiao',
                multi=True,
                placeholder='Select...'
            )
        ], style={'width': '24%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Especialidade'),
            dcc.Dropdown(
                options=[],
                id='filtro_especialidade',
                placeholder='Select...'
            )
        ], style={'width': '24%', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='grafico-linha'),
    dcc.Graph(id='grafico-distribuicao'),
    dcc.Graph(id='grafico-pizza'),
])

# --- Atualiza opções de especialidade ---
@app.callback(
    Output('filtro_especialidade', 'options'),
    Input('filtro_ano', 'value'),
    Input('filtro_mes', 'value'),
    Input('filtro_regiao', 'value'),
)
def atualizar_opcoes_especialidade(ano, mes, regiao):
    dff = df.copy()
    if ano:
        dff = dff[dff['ano'] == ano]
    if mes:
        dff = dff[dff['mes'] == mes]
    if regiao:
        dff = dff[dff['regiao'].isin(regiao)]
    especialidades = sorted(dff['especialidade'].unique())
    return [{'label': i, 'value': i} for i in especialidades]

# --- Atualiza os gráficos ---
@app.callback(
    Output('grafico-linha', 'figure'),
    Output('grafico-distribuicao', 'figure'),
    Output('grafico-pizza', 'figure'),
    Input('filtro_ano', 'value'),
    Input('filtro_mes', 'value'),
    Input('filtro_regiao', 'value'),
    Input('filtro_especialidade', 'value'),
)
def atualizar_dash(ano, mes, regiao, especialidade):
    dff = df.copy()
    if ano:
        dff = dff[dff['ano'] == ano]
    if mes:
        dff = dff[dff['mes'] == mes]
    if regiao:
        dff = dff[dff['regiao'].isin(regiao)]
    if especialidade:
        dff = dff[dff['especialidade'] == especialidade]

    df_linha = dff.groupby(['ano', 'mes'])['qtd_mes_especialidade'].sum().reset_index()
    df_linha['data'] = df_linha['ano'] + '-' + df_linha['mes']
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_linha['data'], y=df_linha['qtd_mes_especialidade'],
        mode='lines+markers', name='Evolução'))
    fig1.update_layout(title='Evolução Temporal',
                       xaxis_title='Data', yaxis_title='Quantidade')

    if 'regiao' in dff.columns:
        graf_dist = px.bar(
            dff,
            x='especialidade', y='qtd_mes_especialidade',
            title='Distribuição de Especialidades por Região',
            color='regiao', barmode='group'
        )
    else:
        graf_dist = px.bar(title='Erro: coluna "regiao" não encontrada.')

    graf_pizza = px.pie(
        dff.groupby('especialidade')['qtd_mes_especialidade'].sum().reset_index(),
        values='qtd_mes_especialidade',
        names='especialidade',
        title='Proporção por Especialidade'
    )

    return fig1, graf_dist, graf_pizza

# --- Executa o servidor ---
if __name__ == '__main__':
    app.run(debug=True)
