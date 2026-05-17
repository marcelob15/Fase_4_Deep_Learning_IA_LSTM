# 1. Atualizado para a versão 3.11-slim
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de dependência e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante dos arquivos (incluindo modelo e scaler gerados)
COPY . .

# 2. Ajustado para a porta 9000 para alinhar com seu comando de run
EXPOSE 9000

# 3. Comando para rodar a aplicação na porta 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]