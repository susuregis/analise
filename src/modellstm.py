import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Caminho para a pasta raiz do projeto (subindo uma pasta a partir de src/)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# 3. Caminho completo para o CSV
csv_path = os.path.join(ROOT_DIR, 'data', 'dados_sih', 'df_final.csv')

# 4. Agora carrega o CSV
df_modelo = pd.read_csv(csv_path)
print(f"Arquivo carregado: {csv_path}")

# 5. Criar coluna 'data' a partir de ano e mês - corrigido para usar nomes esperados pelo pd.to_datetime
df_modelo['data'] = pd.to_datetime(
    df_modelo.rename(columns={'ano': 'year', 'mes': 'month'})
             .assign(day=1)[['year', 'month', 'day']]
)

# 6. Filtrar a especialidade desejada
especialidade = 'Clínicos'
df_lstm = df_modelo[df_modelo['especialidade'] == especialidade][['data', 'qtd_mes_especialidade']].set_index('data')

# 7. Normalizar os dados
scaler = MinMaxScaler()
df_lstm_scaled = scaler.fit_transform(df_lstm)

# 8. Criar sequências temporais
def create_sequences(data, n_steps=3):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

n_steps = 3
X_lstm, y_lstm = create_sequences(df_lstm_scaled, n_steps=n_steps)

# 9. Ajustar o formato para o LSTM
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

# 10. Definir o modelo
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 11. Treinar o modelo
model.fit(X_lstm, y_lstm, epochs=100, verbose=0)

# 12. Fazer previsões
y_pred_lstm = model.predict(X_lstm)

# 13. Reverter a normalização
y_true_inv = scaler.inverse_transform(y_lstm)
y_pred_inv = scaler.inverse_transform(y_pred_lstm)

# 14. Avaliar
rmse_lstm = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
mae_lstm = mean_absolute_error(y_true_inv, y_pred_inv)
mape_lstm = np.mean(np.abs((y_true_inv - y_pred_inv) / y_true_inv)) * 100

print(f"LSTM - {especialidade}:\nRMSE: {rmse_lstm:.2f} | MAE: {mae_lstm:.2f} | MAPE: {mape_lstm:.2f}%")
