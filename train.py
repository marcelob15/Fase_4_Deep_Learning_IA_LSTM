# train.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib

# ==========================================
# 1. Coleta de Dados (Ingestion)
# ==========================================
# Definimos o ticker da empresa (Disney) e a janela temporal
symbol = 'DIS'
start_date = '2018-01-01'
end_date = '2024-07-20'

print(f"Baixando dados para {symbol}...")
df = yf.download(symbol, start=start_date, end=end_date)

# Trava de segurança: impede que o pipeline continue caso a API do Yahoo Finance falhe
if df.empty:
    raise ValueError(f"Falha ao baixar dados de {symbol}. Verifique sua conexão com a internet ou se o Yahoo Finance está respondendo.")

# Extrai apenas a coluna de fechamento ('Close'). O reshape(-1, 1) transforma a série de 1D para uma matriz 2D.
data = df['Close'].values.reshape(-1, 1)


# ==========================================
# 2. Pré-processamento e Engenharia de Features
# ==========================================
# Normalização: Redes Neurais são sensíveis à magnitude dos valores. 
# O MinMaxScaler comprime os preços de dólares para um intervalo seguro entre 0 e 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Definição do tamanho da janela deslizante (look-back period) de 60 dias
sequence_length = 60
x_train, y_train = [], []

# Divisão Temporal (Temporal Train/Test Split): Para séries temporais, separamos os primeiros 80% 
# dos dados cronologicamente para evitar vazamento de dados do futuro para o passado (data leakage).
training_data_len = int(np.ceil(len(data) * .8))
train_data = scaled_data[0:int(training_data_len), :]

# Criando pares ordenados: x_train recebe 60 dias contínuos, y_train recebe o dia 61 (o alvo) 
# prever o valor de fechamento
for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Redimensionamento 3D: A camada LSTM exige entradas no formato (Nº Amostras, Nº Timesteps, Nº Features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# ==========================================
# 3. Arquitetura e Construção do Modelo LSTM
# ==========================================
print("Construindo o modelo LSTM...")
model = Sequential()

# Camada 1: LSTM com 128 neurônios. 'return_sequences=True' passa a série inteira para a próxima camada LSTM.
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Regularização: "Desliga" 20% dos neurônios aleatoriamente para evitar overfitting

# Camada 2: LSTM com 64 neurônios, comprimindo a sequência em uma saída vetorial única
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

# Camada Oculta Tradicional (Dense) para consolidação
model.add(Dense(25))

# Camada de Saída: 1 neurônio representando a previsão de preço contínuo (Regressão)
model.add(Dense(1))

# Otimizador Adam com função de perda de Erro Quadrático Médio (padrão para regressão)
model.compile(optimizer='adam', loss='mean_squared_error')


# ==========================================
# 4. Treinamento da Rede Neural
# ==========================================
print("Iniciando o treinamento (RTX 3060)...")
# Treina o modelo dividindo os dados em lotes (batch_size) de 32 e passando 20 vezes pelo dataset (epochs)
model.fit(x_train, y_train, batch_size=32, epochs=20) 


# ==========================================
# 5. Avaliação do Modelo (Testes)
# ==========================================
# Cria o dataset de testes baseando-se nos 20% finais do histórico original
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
y_test = data[training_data_len:, :] # Os preços reais (não escalados) de teste

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Executando previsões na base de teste
predictions = model.predict(x_test)

# Inversão da escala: converte os valores previstos (0 a 1) de volta para o valor nominal (dólares)
predictions = scaler.inverse_transform(predictions)

# Cálculo de Métricas Finais
# RMSE: Penaliza erros grandes. MAE: Dá a noção exata de desvio médio absoluto.
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
print(f"Métricas de Validação -> RMSE: {rmse:.2f} | MAE: {mae:.2f}")


# ==========================================
# 6. Salvamento e Exportação (Artefatos de Produção)
# ==========================================
print("Salvando o modelo e o scaler...")
# Salva a arquitetura e os pesos aprendidos para não precisar retreinar
model.save('lstm_stock_model.h5')
# Salva as propriedades do scaler (mínimo e máximo observados) para que a API aplique a mesma escala
joblib.dump(scaler, 'scaler.pkl')
print("Treinamento finalizado com sucesso!")