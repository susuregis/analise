# Quantidade de meses para usar como teste
n_test = 6  # por exemplo, últimos 6 meses

# Separar treino e teste
df_train = df_lstm.iloc[:-n_test]
df_test = df_lstm.iloc[-n_test:]

# Normalizar treino e teste separadamente (ideal para evitar vazamento)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(df_train)
test_scaled = scaler.transform(df_test)

# Função para criar sequências (mesmo do seu código)
def create_sequences(data, n_steps=3):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

n_steps = 3

# Criar sequências de treino
X_train, y_train = create_sequences(train_scaled, n_steps)

# Criar sequências de teste
X_test, y_test = create_sequences(test_scaled, n_steps)

# Ajustar formato para LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Agora você pode treinar o modelo com X_train, y_train e avaliar com X_test, y_test
