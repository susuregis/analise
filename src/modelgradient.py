import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Carrega os dados (substitua pelo caminho correto)
df_modelo = pd.read_csv('data/dados_sih/df_final.csv')

# Codifica especialidades
le = LabelEncoder()
df_modelo['especialidade_cod'] = le.fit_transform(df_modelo['especialidade'])

# Features e target
X = df_modelo[['ano', 'mes', 'especialidade_cod']]
y = df_modelo['qtd_mes_especialidade']

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Treinar modelo
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Predições
y_pred = gbr.predict(X_test)

# Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Gradient Boosting:\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")
