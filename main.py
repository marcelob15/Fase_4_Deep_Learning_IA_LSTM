# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import time
import logging
from keras.models import load_model
import joblib
from prometheus_fastapi_instrumentator import Instrumentator

# ==========================================
# 1. Configuração Básica de Logs e FastAPI
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(
    title="API de Previsão de Ações - Tech Challenge",
    description="Predição do valor de fechamento utilizando modelo LSTM com monitoramento Prometheus.",
    version="1.0.0"
)

# ==========================================
# 2. Monitoramento de Performance (Prometheus)
# ==========================================
# A linha abaixo intercepta as requisições da API e gera métricas 
# de latência, uso de memória e total de acessos automaticamente na rota /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ==========================================
# 3. Carregamento dos Artefatos do Modelo
# ==========================================
# Tenta carregar os pesos da rede neural (.h5) e o escalonador (.pkl) gerados pelo train.py
try:
    model = load_model('lstm_stock_model.h5')
    scaler = joblib.load('scaler.pkl')
    logging.info("✅ Modelo e scaler carregados com sucesso.")
except Exception as e:
    logging.critical(f"❌ Erro fatal ao carregar artefatos: {e}")
    model = scaler = None

# ==========================================
# 4. Esquema de Entrada (Input Payload)
# ==========================================
class StockDataInput(BaseModel):
    historical_prices: list[float] = Field(
        ..., 
        description="Lista contendo os exatos últimos 60 preços de fechamento da ação para análise temporal."
    )

    # Esta configuração injeta os 60 valores fictícios diretamente no Swagger UI (/docs)
    # Assim, o avaliador (ou você no vídeo) só precisa clicar em "Try it out" e "Execute"
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "historical_prices": [
                        100.1, 101.2, 100.5, 99.8, 102.0, 101.5, 100.9, 103.1, 102.8, 104.0,
                        103.5, 105.2, 106.1, 105.8, 104.9, 107.0, 106.5, 108.2, 107.9, 109.1,
                        108.5, 110.0, 109.8, 111.2, 110.5, 112.0, 111.8, 113.5, 112.9, 114.1,
                        113.8, 115.0, 114.5, 116.2, 115.8, 117.0, 116.5, 118.1, 117.9, 119.0,
                        118.5, 120.2, 119.8, 121.0, 120.5, 122.1, 121.8, 123.5, 122.9, 124.0,
                        123.5, 125.1, 124.8, 126.0, 125.5, 127.2, 126.8, 128.0, 127.5, 129.1
                    ]
                }
            ]
        }
    }

# ==========================================
# 5. Rotas da API (Endpoints)
# ==========================================

@app.get("/", response_class=HTMLResponse)
def read_root():
    """ Rota raiz (Home) - Fornece uma interface amigável com links úteis. """
    return """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>API LSTM - Tech Challenge Fase 4</title>
        <style>
            body { font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; text-align: center; padding-top: 60px; }
            .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); display: inline-block; max-width: 500px; width: 90%; }
            h1 { color: #0f172a; margin-bottom: 8px; }
            p { color: #475569; margin-bottom: 24px; }
            .btn { display: inline-block; padding: 12px 20px; margin: 8px; text-decoration: none; color: white; border-radius: 8px; font-weight: 500; transition: 0.2s; }
            .btn-docs { background: #3b82f6; }
            .btn-docs:hover { background: #2563eb; }
            .btn-metrics { background: #10b981; }
            .btn-metrics:hover { background: #059669; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 API de Previsão de Ações (LSTM)</h1>
            <p>Tech Challenge - Fase 04 | MLET</p>
            <a href="/docs" class="btn btn-docs">📖 Swagger UI</a>
            <a href="/metrics" class="btn btn-metrics">📊 Métricas Prometheus</a>
        </div>
    </body>
    </html>
    """

@app.post("/predict")
def predict_stock_price(data: StockDataInput):
    """ Recebe o histórico de fechamentos e retorna o preço previsto para o dia seguinte. """
    
    # Validação de segurança de modelo e payload
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo ou scaler não carregados. Verifique os arquivos .h5 e .pkl na raiz.")

    if len(data.historical_prices) != 60:
        raise HTTPException(status_code=400, detail="O payload deve conter exatamente 60 valores históricos.")

    # Inicia a contagem de tempo de inferência
    start_time = time.time()
    
    try:
        # Passo A: Transforma a lista numa matriz 2D e aplica o escalonamento (0 a 1)
        input_data = np.array(data.historical_prices).reshape(-1, 1)
        scaled_input = scaler.transform(input_data)
        
        # Passo B: Redimensiona para 3D (Samples: 1, Time Steps: 60, Features: 1) esperado pela LSTM
        X = np.reshape(scaled_input, (1, 60, 1))
        
        # Passo C: Inferência do modelo (verbose=0 para ocultar logs do Keras no terminal)
        predicted_scaled = model.predict(X, verbose=0)
        
        # Passo D: Desfaz a normalização para trazer o valor de volta à escala original (dólares)
        predicted_price = float(scaler.inverse_transform(predicted_scaled)[0][0])
        
        # Calcula o tempo total que o modelo levou para responder em milissegundos
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "predicted_close_price": round(predicted_price, 2),
            "model_used": "LSTM",
            "inference_time_ms": round(latency_ms, 2)
        }
        
    except Exception as e:
        logging.error(f"❌ Erro na inferência: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Falha interna na predição: {str(e)}")