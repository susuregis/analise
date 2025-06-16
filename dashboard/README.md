# Dashboard de Análise de Internações SUS

Esta pasta contém os dashboards para visualização e análise dos dados de internações do SUS.

## Arquivos

1. `applight.py` - **Dashboard Leve**: Versão otimizada do dashboard com recursos essenciais, ideal para ambientes com recursos limitados. Esta é a versão recomendada para uso geral.

2. `app.py` - **Dashboard Completo**: Versão completa do dashboard com todas as funcionalidades, como análises avançadas, previsões LSTM e visualizações detalhadas.

## Como Executar

### Versão Básica

Para executar o dashboard leve (sem previsões LSTM), use o script `iniciar_dashboard_leve.bat` na raiz do projeto:

```
cd C:\Users\55819\Downloads\projeto_pyspark
iniciar_dashboard_leve.bat
```

Ou execute diretamente:

```
python dashboard/applight.py
```

### Versão com Previsões LSTM

Para executar o dashboard com suporte a previsões LSTM:

1. Configure o ambiente primeiro (apenas uma vez):

```
configurar_ambiente_lstm.bat
```

2. Inicie o dashboard com previsões LSTM:

```
iniciar_dashboard_lstm.bat
```

## Dados

O dashboard utiliza os dados do arquivo `df_final.csv` localizado em `data/dados_sih/`.

## Funcionalidades

- Visualização de internações por especialidade e região
- Análise temporal de dados históricos
- Previsões LSTM (quando disponíveis)
- Métricas de desempenho do modelo
- Filtros por ano, região e especialidade

## Requisitos para Previsões LSTM

Para habilitar as previsões LSTM, você precisa:

1. Python 3.8 - 3.11
2. TensorFlow 2.15.0 
3. scikit-learn
4. O arquivo do modelo em `src/meu_modelo_lstm.h5`

Use o script `configurar_ambiente_lstm.bat` para configurar automaticamente o ambiente.
