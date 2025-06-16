import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dois níveis acima do arquivo .py
csv_path = os.path.join(BASE_DIR, 'data', 'dados_sih', 'df_final.csv')
df_modelo = pd.read_csv(csv_path)


# 2. Criar coluna 'data' a partir de ano e mês
df_modelo['data'] = pd.to_datetime(
    df_modelo[['ano', 'mes']].rename(columns={'ano': 'year', 'mes': 'month'}).assign(day=1)
)


# 3. Filtrar especialidade desejada
especialidade = 'Clínicos'
df_lstm = df_modelo[df_modelo['especialidade'] == especialidade][['data', 'qtd_mes_especialidade']].set_index('data')

# 4. Ordenar os dados por data (importante para séries temporais)
df_lstm = df_lstm.sort_index()

# 5. Normalizar os dados
scaler = MinMaxScaler()
df_lstm_scaled = scaler.fit_transform(df_lstm)

# 6. Criar sequências temporais (janelas)
def create_sequences(data, n_steps=3):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

n_steps = 3
X, y = create_sequences(df_lstm_scaled, n_steps=n_steps)

# 7. Dividir treino/teste temporal
# Exemplo: últimos 6 meses como teste
test_size = 6
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# 8. Ajustar formato para LSTM (amostras, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 9. Definir o modelo
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 10. Treinar o modelo
model.fit(X_train, y_train, epochs=100, verbose=0)

# 11. Fazer previsões
y_pred = model.predict(X_test)

# 12. Inverter escala
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# 13. Avaliar modelo
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

print(f"Teste:\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

model.save('meu_modelo_lstm.h5')