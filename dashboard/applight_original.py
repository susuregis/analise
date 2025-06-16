import os
from datetime import datetime
import traceback
import sys

# Definir variáveis de ambiente para o TensorFlow antes de importar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprimir avisos do TensorFlow

# Tentar importar as bibliotecas necessárias
try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output
    import pandas as pd
    import plotly.graph_objs as go
    import numpy as np
    print("✓ Dependências básicas carregadas com sucesso")
      # Tentar importar TensorFlow e sklearn (opcional)
    tensorflow_available = False
    sklearn_available = False
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        tensorflow_available = True
        print("✓ TensorFlow carregado com sucesso")
    except ImportError as e:
        print(f"⚠️ TensorFlow não disponível: {e}")
        print("As previsões LSTM serão desabilitadas")
        
    try:
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        sklearn_available = True
        print("✓ Scikit-learn carregado com sucesso")
    except ImportError as e:
        print(f"⚠️ Scikit-learn não disponível: {e}")
        print("As previsões LSTM serão desabilitadas")
        
except ImportError as e:
    print(f"ERRO: Não foi possível importar todas as bibliotecas necessárias: {e}")
    print("Por favor, instale as bibliotecas necessárias usando o comando:")
    print("pip install dash pandas plotly")
    sys.exit(1)

# Estilo
COLORS = {
    'primary': '#4CAF50',
    'secondary': '#2196F3',
    'accent': '#FF9800',
    'text': '#333333',
    'background': '#f9f9f9'
}

app_style = {
    'font-family': 'sans-serif',
    'background-color': COLORS['background'],
    'padding': '20px',
    'color': COLORS['text'],
    'max-width': '1400px',
    'margin': '0 auto'
}

card_style = {
    'padding': '15px',
    'border-radius': '10px',
    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'margin-bottom': '20px',
    'background-color': 'white'
}

# Carregar dados
try:
    print("Carregando dados...")
    df = pd.read_csv(r'C:\Users\55819\Downloads\projeto_pyspark\data\dados_sih\df_final.csv')
    
    # Criar coluna 'data' de forma segura para evitar erros de manipulação de datas
    # Convertendo para strings de data primeiro e depois para datetime
    df['data'] = pd.to_datetime(
        df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01', 
        format='%Y-%m-%d'
    )
    print(f"Dados carregados: {len(df)} registros")
except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    df = pd.DataFrame()  # DataFrame vazio como fallback

# --- Tentar carregar modelo LSTM ---
model = None
n_steps = 3  # Passos usados na LSTM
use_lstm = tensorflow_available and sklearn_available

if use_lstm:
    try:
        print("Tentando carregar o modelo LSTM...")
        modelo_caminho = r'C:\Users\55819\Downloads\projeto_pyspark\src\meu_modelo_lstm.h5'
        if os.path.exists(modelo_caminho):
            model = load_model(modelo_caminho, compile=False)
            
            # Obter estatísticas do modelo
            modelo_data_criacao = datetime.fromtimestamp(os.path.getmtime(modelo_caminho)).strftime('%d/%m/%Y')
            modelo_tamanho = round(os.path.getsize(modelo_caminho) / (1024 * 1024), 2)  # Tamanho em MB
            
            print(f"✓ Modelo LSTM carregado com sucesso! ({modelo_tamanho} MB - criado em {modelo_data_criacao})")
        else:
            print(f"⚠️ AVISO: Arquivo do modelo não encontrado em: {modelo_caminho}")
            print("As previsões LSTM estarão desabilitadas")
            use_lstm = False
    except Exception as e:
        print(f"⚠️ Erro ao carregar o modelo LSTM: {e}")
        print("As previsões LSTM estarão desabilitadas")
        use_lstm = False
else:
    print("⚠️ Requisitos para LSTM não disponíveis - previsões desabilitadas")

# Criar app
app = dash.Dash(__name__)
app.title = "Dashboard de Análise de Internações SUS (Versão Leve)"

# Layout simplificado
app.layout = html.Div([
    # Cabeçalho
    html.Div([
        html.H1("Dashboard de Análise de Internações SUS (Versão Leve)", 
                style={'textAlign': 'center', 'color': COLORS['primary'], 'margin-bottom': '10px'}),
        html.P("Análise temporal de internações por especialidade e região", 
               style={'textAlign': 'center', 'margin-bottom': '30px', 'font-size': '16px'}),
        
        html.Div([
            html.P("⚠️ NOTA: Esta é uma versão com funcionalidades reduzidas devido a limitações de espaço em disco.",
                   style={'textAlign': 'center', 'color': COLORS['accent'], 'font-weight': 'bold'}),
            html.P("Para habilitar previsões e análises avançadas, libere espaço em disco e reinicie o aplicativo.",
                   style={'textAlign': 'center'})
        ], style={'background-color': '#FFF3E0', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '20px'})
    ]),
    
    # Filtros
    html.Div([
        html.Div([
            html.H3("Filtros de Análise", style={'color': COLORS['secondary']}),
            
            html.Label("Ano", style={'font-weight': 'bold', 'margin-top': '10px'}),
            dcc.Dropdown(
                id='filtro-ano',
                options=[{"label": str(ano), "value": ano} for ano in sorted(df['ano'].unique())],
                multi=True,
                placeholder="Selecione o(s) ano(s)"
            ),

            html.Label("Região", style={'font-weight': 'bold', 'margin-top': '10px'}),
            dcc.Dropdown(
                id='filtro-regiao',
                options=[{"label": r, "value": r} for r in sorted(df['regiao'].unique())],
                multi=True,
                placeholder="Selecione a(s) região(ões)"
            ),
            
            html.Label("Especialidade", style={'font-weight': 'bold', 'margin-top': '10px'}),
            dcc.Dropdown(
                id='filtro-especialidade',
                options=[{"label": e, "value": e} for e in sorted(df['especialidade'].unique())],
                multi=True,
                placeholder="Selecione a(s) especialidade(s)"
            ),
        ], style=card_style),
    ]),
    
    # Gráficos simplificados    html.Div([
        html.H3("Distribuição por Especialidade", style={'color': COLORS['secondary']}),
        dcc.Graph(id='grafico-barras')
    ], style=card_style),
    
    html.Div([
        html.H3("Evolução Temporal", style={'color': COLORS['secondary']}),
        dcc.Graph(id='grafico-timeline')
    ], style=card_style),
    
    # Seção de previsões LSTM - visível apenas se o modelo estiver disponívelhtml.Div([
    html.H3("📈 Previsões LSTM - Próximos 12 meses", 
                style={'color': COLORS['primary'], 'textAlign': 'center'}),
        html.Div([
            html.P("Previsões baseadas no modelo de aprendizado profundo (LSTM)", 
                style={'textAlign': 'center', 'fontStyle': 'italic', 'marginBottom': '5px'}),
            html.P([
                "Modelo criado em: ", 
                html.Span(
                    modelo_data_criacao if 'modelo_data_criacao' in locals() else "N/A", 
                    style={'fontWeight': 'bold'}
                ),
                " | Dados de treinamento: ",
                html.Span(
                    f"{df['data'].min().strftime('%m/%Y') if not df.empty else 'N/A'} a {df['data'].max().strftime('%m/%Y') if not df.empty else 'N/A'}", 
                    style={'fontWeight': 'bold'}
                )
            ], style={'textAlign': 'center', 'fontSize': '13px', 'marginBottom': '15px', 'color': '#666'})
        ]),
        dcc.Graph(id='grafico-previsao'),
        
        html.Div([
            html.H4("Métricas do Modelo", style={'color': COLORS['secondary'], 'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Label("RMSE:", style={'fontWeight': 'bold'}),
                    html.Span(id='modelo-rmse')
                ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.Label("MAE:", style={'fontWeight': 'bold'}),
                    html.Span(id='modelo-mae')
                ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.Label("MAPE:", style={'fontWeight': 'bold'}),
                    html.Span(id='modelo-mape')
                ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'})
            ])
        ], style={'marginTop': '15px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
    ], style=dict(card_style, **{'display': 'block' if use_lstm else 'none'}), id='secao-previsoes'),
    
    # Mensagem quando o LSTM não está disponível
    html.Div([
        html.H3("⚠️ Previsões LSTM não disponíveis", 
                style={'color': COLORS['accent'], 'textAlign': 'center'}),
        html.P("O modelo de previsão LSTM não está disponível porque:", 
               style={'textAlign': 'center'}),
        html.Ul([
            html.Li("As dependências TensorFlow/Scikit-learn não estão instaladas, ou"),
            html.Li("O arquivo do modelo não foi encontrado, ou"),
            html.Li("Houve um erro ao carregar o modelo")
        ], style={'width': '80%', 'margin': '0 auto', 'textAlign': 'left'}),
        html.P("Para ativar as previsões, instale as dependências necessárias:", 
               style={'textAlign': 'center', 'marginTop': '15px'}),
        html.Pre("pip install tensorflow scikit-learn",
                style={'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '5px', 'width': '80%', 'margin': '10px auto'})
    ], style=dict(card_style, **{'display': 'block' if not use_lstm else 'none'}), id='mensagem-sem-modelo'),
    
    # Tabela de dados
    html.Div([
        html.H3("Dados Detalhados", style={'color': COLORS['secondary']}),
        dash_table.DataTable(
            id='tabela-dados',
            columns=[
                {'name': 'Ano', 'id': 'ano'},
                {'name': 'Mês', 'id': 'mes'},
                {'name': 'Região', 'id': 'regiao'},
                {'name': 'Especialidade', 'id': 'especialidade'},
                {'name': 'Quantidade', 'id': 'qtd_mes_especialidade'},
            ],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': COLORS['secondary'],
                'color': 'white',
                'fontWeight': 'bold'
            }
        )
    ], style=card_style),
    
    # Rodapé
    html.Footer([
        html.P(f"Dashboard atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
               style={'text-align': 'center', 'margin-top': '40px', 'color': '#777'}),
        html.P("Versão leve - Recursos limitados",
               style={'text-align': 'center', 'color': '#777'})
    ])
], style=app_style)

# Configurar retentativas de callback para evitar erros
app.config.suppress_callback_exceptions = True

# Callback
@app.callback(
    [Output('grafico-barras', 'figure'),
     Output('grafico-timeline', 'figure'),
     Output('grafico-previsao', 'figure'),
     Output('tabela-dados', 'data'),
     Output('modelo-rmse', 'children'),
     Output('modelo-mae', 'children'),
     Output('modelo-mape', 'children'),
     Output('secao-previsoes', 'style'),
     Output('mensagem-sem-modelo', 'style')],
    [Input('filtro-ano', 'value'),
     Input('filtro-regiao', 'value'),
     Input('filtro-especialidade', 'value')],
    # Prevenir atualizações iniciais em branco
    prevent_initial_call=False
)
def atualizar_graficos(f_ano, f_regiao, f_especialidade):
    import plotly.graph_objs as go
    
    try:
        df_filtro = df.copy()
        
        # Aplicar filtros se não forem None
        if f_ano:
            df_filtro = df_filtro[df_filtro['ano'].isin(f_ano)]
        if f_regiao:
            df_filtro = df_filtro[df_filtro['regiao'].isin(f_regiao)]
        if f_especialidade:
            df_filtro = df_filtro[df_filtro['especialidade'].isin(f_especialidade)]
        
        # Configurar visibilidade das seções LSTM
        secao_previsoes_style = dict(card_style, **{'display': 'block' if use_lstm else 'none'})
        mensagem_sem_modelo_style = dict(card_style, **{'display': 'block' if not use_lstm else 'none'})
        
        # Valores default para métricas
        rmse_valor = "N/A"
        mae_valor = "N/A"
        mape_valor = "N/A"
        
        # Garantir que temos dados para trabalhar
        if df_filtro.empty:
            # Criar visualizações vazias se não houver dados
            fig_barras = go.Figure()
            fig_barras.update_layout(title="Sem dados para exibir com estes filtros")
            
            fig_timeline = go.Figure()
            fig_timeline.update_layout(title="Sem dados para exibir com estes filtros")
            
            fig_previsao = go.Figure()
            fig_previsao.update_layout(title="Sem dados para previsão")
            
            return fig_barras, fig_timeline, fig_previsao, [], rmse_valor, mae_valor, mape_valor, secao_previsoes_style, mensagem_sem_modelo_style
        
        # Gráfico de barras
        barras = df_filtro.groupby('especialidade')['qtd_mes_especialidade'].sum().reset_index()
        barras = barras.sort_values('qtd_mes_especialidade', ascending=False)
        
        fig_barras = go.Figure()
        fig_barras.add_trace(go.Bar(
            x=barras['especialidade'],
            y=barras['qtd_mes_especialidade'],
            marker_color=COLORS['primary'],
            text=barras['qtd_mes_especialidade'].round(0).astype(int),
            textposition='auto'
        ))
        
        fig_barras.update_layout(
            xaxis_title='Especialidade',
            yaxis_title='Quantidade',
            plot_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            )
        )
        
        # Gráfico timeline
        timeline = df_filtro.groupby(['data', 'especialidade'])['qtd_mes_especialidade'].sum().reset_index()
        
        # Se houver muitas especialidades, limitar às 5 maiores para melhorar a visualização
        top_especialidades = []
        if len(timeline['especialidade'].unique()) > 5:
            top_especialidades = df_filtro.groupby('especialidade')['qtd_mes_especialidade'].sum().nlargest(5).index.tolist()
            timeline = timeline[timeline['especialidade'].isin(top_especialidades)]
        else:
            top_especialidades = timeline['especialidade'].unique().tolist()
        
        fig_timeline = go.Figure()
        for especialidade in timeline['especialidade'].unique():
            df_esp = timeline[timeline['especialidade'] == especialidade]
            if not df_esp.empty:  # Verificar se temos dados para esta especialidade
                fig_timeline.add_trace(go.Scatter(
                    x=df_esp['data'],
                    y=df_esp['qtd_mes_especialidade'],
                    mode='lines+markers',
                    name=especialidade,
                    hovertemplate='%{x|%b %Y}: %{y:,.0f}<extra>' + especialidade + '</extra>'
                ))
        
        fig_timeline.update_layout(
            xaxis_title='Data',
            yaxis_title='Quantidade',
            plot_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5
            )
        )
        
        # Criar gráfico de previsão LSTM
        fig_previsao = go.Figure()
        
        # Verificar se podemos fazer previsões
        if use_lstm and model is not None:
            # Preparar dados para previsão
            rmse_values = []
            mae_values = []
            mape_values = []
            
            # Cores para as especialidades
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
            
            # Fazer previsões para cada especialidade
            for idx, especialidade in enumerate(top_especialidades):
                if idx >= 5:  # Limitar a 5 especialidades
                    break
                    
                try:
                    # Filtrar dados para esta especialidade
                    df_esp = df_filtro[df_filtro['especialidade'] == especialidade]
                    df_esp = df_esp.groupby('data')['qtd_mes_especialidade'].sum().sort_index()
                    
                    if len(df_esp) < n_steps:
                        continue  # Pular se não tiver dados suficientes
                        
                    # Normalizar os dados
                    scaler = MinMaxScaler()
                    serie_scaled = scaler.fit_transform(df_esp.values.reshape(-1, 1))
                    
                    # Criar sequências para validação
                    X_test, y_test = [], []
                    for i in range(n_steps, len(serie_scaled)):
                        X_test.append(serie_scaled[i-n_steps:i])
                        y_test.append(serie_scaled[i])
                    
                    if not X_test:  # Verificar se temos dados de teste
                        continue
                        
                    X_test = np.array(X_test).reshape(-1, n_steps, 1)
                    y_test = np.array(y_test)
                    
                    # Fazer previsões em dados de teste
                    y_pred = model.predict(X_test, verbose=0)
                    
                    # Calcular métricas (valores normalizados)
                    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
                    mae = np.mean(np.abs(y_pred - y_test))
                    
                    # Desnormalizar para MAPE
                    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                      # Calcular MAPE (evitando divisão por zero e outliers)
                    mask = y_test_inv > 1.0  # Evitar valores muito pequenos que distorcem o MAPE
                    if np.any(mask):
                        # Calcular erro percentual absoluto para cada ponto
                        absolute_percentage_errors = np.abs((y_test_inv[mask] - y_pred_inv[mask]) / y_test_inv[mask]) * 100
                        # Limitar outliers (erros extremos podem distorcer a média)
                        capped_errors = np.clip(absolute_percentage_errors, 0, 100)
                        # Média dos erros percentuais absolutos
                        mape = np.mean(capped_errors)
                    else:
                        mape = 0
                    
                    # Guardar métricas
                    rmse_values.append(rmse)
                    mae_values.append(mae)
                    mape_values.append(mape)
                    
                    # Previsões futuras
                    entrada = serie_scaled[-n_steps:].reshape(1, n_steps, 1)
                    preds = []
                    
                    # Gerar previsões para os próximos 12 meses
                    for _ in range(12):
                        pred = model.predict(entrada, verbose=0)[0][0]
                        preds.append(pred)
                        entrada = np.append(entrada[:, 1:, :], [[[pred]]], axis=1)
                      # Desnormalizar as previsões
                    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                    
                    # Criar datas futuras - usando pd.DateOffset para evitar erros com Timestamp
                    try:
                        ultima_data = df_esp.index[-1]
                        # Criar data inicial de forma mais robusta
                        if isinstance(ultima_data, pd.Timestamp):
                            data_inicial = pd.Timestamp(ultima_data.year, ultima_data.month, 1) + pd.DateOffset(months=1)
                        else:
                            # Fallback se o índice não for datetime
                            data_atual = pd.Timestamp.now()
                            data_inicial = pd.Timestamp(data_atual.year, data_atual.month, 1) + pd.DateOffset(months=1)
                            
                        datas_futuras = pd.date_range(start=data_inicial, periods=12, freq='MS')
                    except Exception as e:
                        print(f"Erro ao criar datas futuras: {e}")
                        # Fallback: criar 12 meses a partir de agora
                        data_atual = pd.Timestamp.now()
                        data_inicial = pd.Timestamp(data_atual.year, data_atual.month, 1) + pd.DateOffset(months=1)
                        datas_futuras = pd.date_range(start=data_inicial, periods=12, freq='MS')
                    
                    # Cor para esta especialidade
                    cor_linha = colors[idx % len(colors)]
                    
                    # Adicionar dados históricos
                    fig_previsao.add_trace(go.Scatter(
                        x=df_esp.index,
                        y=df_esp.values,
                        mode='lines',
                        name=f"{especialidade} (histórico)",
                        line=dict(color=cor_linha),
                        hovertemplate='%{x|%b %Y}: %{y:,.0f}<extra>' + especialidade + ' (histórico)</extra>'
                    ))
                    
                    # Adicionar previsões
                    fig_previsao.add_trace(go.Scatter(
                        x=datas_futuras,
                        y=preds_inv,
                        mode='lines+markers',
                        name=f"{especialidade} (previsão)",
                        line=dict(color=cor_linha, dash='dash'),
                        marker=dict(symbol='circle', size=8),
                        hovertemplate='%{x|%b %Y}: %{y:.0f}<extra>' + especialidade + ' (previsão)</extra>'
                    ))                    # Intervalo de confiança (simples, 10%)
                    try:
                        # Calcular intervalo baseado tanto na previsão quanto na variabilidade histórica
                        historico_std = df_esp.values.std() / df_esp.values.mean() if df_esp.values.mean() > 0 else 0.1
                        # Ajustar dinamicamente o intervalo de confiança (entre 5% e 20%)
                        variabilidade = min(max(historico_std * 100, 5), 20) / 100
                        
                        upper_bound = preds_inv * (1 + variabilidade)
                        lower_bound = preds_inv * (1 - variabilidade)
                        # Garantir que o limite inferior não seja negativo
                        lower_bound = np.maximum(lower_bound, np.zeros_like(lower_bound))
                    except Exception:
                        # Fallback para 10% fixo se algo der errado
                        upper_bound = preds_inv * 1.1
                        lower_bound = preds_inv * 0.9
                        lower_bound = np.maximum(lower_bound, np.zeros_like(lower_bound))
                    
                    # Adicionar área de intervalo de confiança
                    # Converter cor hexadecimal para rgba para transparência
                    def hex_to_rgba(hex_color, alpha=0.1):
                        try:
                            # Remover o # se presente
                            hex_color = hex_color.lstrip('#')
                            # Converter para valores RGB
                            if len(hex_color) == 6:
                                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                                return f'rgba({r}, {g}, {b}, {alpha})'
                            elif len(hex_color) == 3:  # Suporte para formato curto, ex: #ABC
                                r, g, b = int(hex_color[0], 16) * 17, int(hex_color[1], 16) * 17, int(hex_color[2], 16) * 17
                                return f'rgba({r}, {g}, {b}, {alpha})'
                            return f'rgba(128, 128, 128, {alpha})' # Fallback para cinza
                        except Exception:
                            return f'rgba(128, 128, 128, {alpha})'  # Fallback seguro
                    
                    fill_color = hex_to_rgba(cor_linha, 0.1)
                    
                    fig_previsao.add_trace(go.Scatter(
                        x=list(datas_futuras) + list(datas_futuras)[::-1],
                        y=list(upper_bound) + list(lower_bound)[::-1],
                        fill='toself',
                        fillcolor=fill_color,
                        line=dict(color='rgba(0,0,0,0)'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                    
                except Exception as e:
                    print(f"Erro ao fazer previsão para {especialidade}: {str(e)}")
                    continue
            
            # Adicionar linha vertical separando histórico e previsão
            try:
                if len(df_esp) > 0:
                    # Usar a última data observada como separação entre histórico e previsão
                    ultima_data_hist = df_esp.index[-1]
                    fig_previsao.add_vline(
                        x=ultima_data_hist,
                        line_width=1, line_dash="dash", line_color="gray",
                        annotation_text="Início previsão",
                        annotation_position="top right"
                    )
            except Exception as e:
                print(f"Erro ao adicionar linha vertical de separação: {str(e)}")
            
            # Calcular médias das métricas
            if rmse_values:
                rmse_valor = f"{np.mean(rmse_values):.4f}"
                mae_valor = f"{np.mean(mae_values):.4f}"
                mape_valor = f"{np.mean(mape_values):.2f}%"
            
            fig_previsao.update_layout(
                title="Previsão de Internações por Especialidade (LSTM)",
                xaxis_title="Data",
                yaxis_title="Quantidade de Internações",
                plot_bgcolor='white',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.2,
                    xanchor='center',
                    x=0.5
                )
            )
        else:
            # Se não temos previsão, mostrar gráfico vazio
            fig_previsao.update_layout(
                title="Previsões LSTM não disponíveis",
                annotations=[dict(
                    text="Modelo LSTM não disponível ou dependências faltando",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]            )
        
        # Tabela
        dados_tabela = df_filtro.to_dict('records')
        return fig_barras, fig_timeline, fig_previsao, dados_tabela, rmse_valor, mae_valor, mape_valor, secao_previsoes_style, mensagem_sem_modelo_style
        
    except Exception as e:
        # Em caso de erro, exibir gráficos de erro e logar o problema
        import traceback
        import datetime
        
        # Obter informações detalhadas do erro
        error_trace = traceback.format_exc()
        error_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_summary = f"Erro ({error_time}): {str(e)}"
        
        print(f"\n{'='*50}\n{error_summary}\n{'='*50}")
        print(error_trace)
        
        # Log do erro em um arquivo para diagnóstico posterior
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"error_log_{datetime.datetime.now().strftime('%Y%m%d')}.txt"), 'a') as log_file:
                log_file.write(f"\n{'='*50}\n{error_summary}\n{error_trace}\n{'='*50}\n")
        except:
            print("Não foi possível salvar o log de erros")
        
        # Criar gráficos vazios com mensagem amigável
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Ocorreu um erro ao processar dados",
            plot_bgcolor='white',
            annotations=[dict(
                text=f"Erro: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(size=14, color='#721c24')
            )]
        )
        
        # Mensagem adicional para ajudar o usuário
        sugestao_fig = go.Figure()
        sugestao_fig.update_layout(
            title="Sugestão",
            plot_bgcolor='white',
            annotations=[dict(
                text="Tente aplicar filtros diferentes ou reiniciar o aplicativo",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(size=14)
            )]
        )
        
        # Configurar visibilidade das seções LSTM
        secao_previsoes_style = dict(card_style, **{'display': 'block' if use_lstm else 'none'})
        mensagem_sem_modelo_style = dict(card_style, **{'display': 'block' if not use_lstm else 'none'})
        
        return empty_fig, empty_fig, sugestao_fig, [], "N/A", "N/A", "N/A", secao_previsoes_style, mensagem_sem_modelo_style

if __name__ == '__main__':
    print("Iniciando Dashboard de Análise de Internações SUS (Versão Leve)...")
    print(f"Acesse o dashboard em http://127.0.0.1:8050/")
    try:
        # Usa tratamento de erro para garantir graceful shutdown
        app.run(debug=True, dev_tools_hot_reload=False)
    except KeyboardInterrupt:
        print("Encerrando o servidor...")
    except Exception as e:
        print(f"Erro ao iniciar o servidor: {str(e)}")
        print("Certifique-se de que a porta 8050 não está em uso.")
        print("Tentando iniciar em uma porta alternativa...")
        try:
            app.run(debug=True, port=8051, dev_tools_hot_reload=False)
            print("Servidor iniciado na porta 8051: http://127.0.0.1:8051/")
        except Exception as e2:
            print(f"Não foi possível iniciar o servidor: {str(e2)}")
