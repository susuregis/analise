# Projeto de Análise e Previsão de Internações SUS

Este projeto consiste em um sistema de análise de dados e previsão de internações do Sistema Único de Saúde (SUS) brasileiro, utilizando técnicas de Machine Learning para realizar previsões de séries temporais.

## Estrutura do Projeto

```
├── autent/              # Sistema de autenticação
├── dashboard/           # Aplicações de visualização de dados
│   ├── app.py           # Dashboard completo com todas as funcionalidades
│   └── applight.py      # Versão leve do dashboard
├── data/
│   └── dados_sih/       # Dados do Sistema de Informações Hospitalares
│       ├── df_final.csv # Dataset principal processado
│       └── ...          # Outros arquivos de dados
├── src/                 # Códigos-fonte dos modelos
│   ├── meu_modelo_lstm.h5  # Modelo LSTM treinado
│   ├── modellstm.py     # Implementação do modelo LSTM
│   ├── modelgradient.py # Implementação do modelo Gradient Boosting
│   └── treino.py        # Script de treinamento dos modelos
```

## Desenvolvimento do Modelo de Machine Learning

### Abordagem
O projeto utiliza redes neurais LSTM (Long Short-Term Memory) para previsão de séries temporais de internações hospitalares. Este tipo de modelo é especialmente adequado para capturar padrões temporais em séries de dados.

### Preparação dos Dados
1. **Carregamento e Filtragem**: Os dados são carregados do arquivo `df_final.csv` e filtrados por especialidade médica (ex: "Clínicos").
2. **Criação de Features Temporais**: Uma coluna de data é criada a partir das colunas de ano e mês.
3. **Normalização**: Os dados são normalizados utilizando `MinMaxScaler` para otimizar o treinamento do modelo.
4. **Criação de Sequências**: São criadas janelas temporais de 3 passos anteriores (n_steps=3) para prever o próximo valor.

### Arquitetura do Modelo
```python
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
```

O modelo consiste em:
- Uma camada LSTM com 50 unidades e função de ativação ReLU
- Uma camada Dense de saída para previsão de um único valor



### Avaliação do Modelo
O desempenho é avaliado através de múltiplas métricas:
- **RMSE** (Root Mean Squared Error): Erro quadrático médio da raiz
- **MAE** (Mean Absolute Error): Erro absoluto médio
- **MAPE** (Mean Absolute Percentage Error): Erro percentual absoluto médio

## Dados Armazenados

### Fontes de Dados
Os dados utilizados são do Sistema de Informações Hospitalares (SIH) do SUS, complementados com dados demográficos do IBGE.

### Estrutura dos Dados
O arquivo principal `df_final.csv` contém:

- **Dimensões Temporais**:
  - `ano`: Ano da internação
  - `mes`: Mês da internação
  - `data`: Data formatada (criada no processamento)

- **Dimensões Geográficas**:
  - `regiao`: Região do Brasil (Norte, Nordeste, Centro-Oeste, Sudeste, Sul)

- **Dimensões de Especialidade**:
  - `especialidade`: Especialidade médica da internação (ex: Clínicos, Cirúrgicos, etc.)

- **Métricas**:
  - `qtd_mes_especialidade`: Quantidade de internações por mês e especialidade

### Arquivos Auxiliares
- `Cities_Brazil_IBGE.xlsx`: Dados demográficos das cidades brasileiras
- `Internações_por_Região_segundo_Ano_atendimento.csv`: Dados agregados por região e ano
- `Quantidade_existente_por_Especialidade_segundo_Anomês_compet.csv`: Dados por especialidade

## Como Executar o Projeto

### Requisitos
- Python 3.8 - 3.11
- Dependências listadas em `requirements.txt`

### Dashboard Versão Leve (Sem Previsões LSTM)
```bash
python dashboard/applight.py
```

### Dashboard Versão Completa (Com Previsões LSTM)
```bash
# Instalar dependências necessárias
pip install tensorflow==2.15.0 scikit-learn

# Executar o dashboard completo
python dashboard/app.py
```

## Funcionalidades
- Visualização de internações por especialidade e região
- Análise temporal de dados históricos
- Previsões futuras usando modelo LSTM
- Métricas de desempenho do modelo
- Filtros interativos por região, especialidade e período

---

Este projeto demonstra como técnicas de Machine Learning podem ser aplicadas para análise e previsão de dados de saúde pública, permitindo melhores insights para gestores e pesquisadores da área.
