# 🚀 Tech Challenge Fase 4 — Machine Learning Engineering

**Nome:** Marcelo Bertin  
**Matrícula:** RM 368902  

---

## 🎥 Demonstração do Projeto

Apresentação do projeto no YouTube:

🔗 https://www.youtube.com/watch?v=gkpc7WcL2Uw

---

## 📌 Sobre o Projeto

Este repositório contém a solução desenvolvida para o **Tech Challenge — Fase 4**, com foco em:

- Deep Learning
- Machine Learning Engineering
- APIs de Inferência
- MLOps
- Observabilidade
- Deploy em Produção

O projeto implementa uma solução completa de previsão de séries temporais financeiras utilizando redes neurais **LSTM (Long Short-Term Memory)** para prever o valor de fechamento das ações da **Disney (DIS)**.

A aplicação contempla todo o ciclo de Machine Learning Engineering:

- 📈 Coleta automatizada de dados financeiros
- 🧠 Treinamento e validação do modelo
- 🌐 API REST com FastAPI
- 📊 Monitoramento com Prometheus
- 🐳 Containerização com Docker
- ⚙️ Estrutura preparada para deploy escalável

---

# 📋 Objetivo do Projeto

O objetivo principal é desenvolver um sistema capaz de:

- Coletar dados históricos financeiros
- Processar e normalizar séries temporais
- Treinar um modelo de Deep Learning
- Realizar previsões futuras de preços
- Disponibilizar inferências via API REST
- Monitorar a aplicação em ambiente de produção

A solução segue práticas modernas de:

- Machine Learning Engineering
- MLOps
- APIs de Inferência
- Observabilidade
- Deploy Containerizado

---

# ✅ Requisitos do Projeto Atendidos

## 📥 Coleta e Pré-processamento

- Utilização da biblioteca `yfinance`
- Obtenção automática de dados históricos
- Limpeza e organização dos dados financeiros
- Normalização com `MinMaxScaler`
- Preparação de janelas temporais para entrada da LSTM

---

## 🧠 Modelo de Deep Learning

Implementação de rede neural recorrente baseada em:

- LSTM (Long Short-Term Memory)
- Camadas Dense
- Camadas Dropout para redução de overfitting

O modelo foi projetado para identificar padrões temporais e tendências financeiras.

---

## 📊 Avaliação do Modelo

O treinamento utiliza métricas clássicas de regressão:

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

Essas métricas permitem validar a qualidade das previsões realizadas.

---

## 🌐 Deploy da Aplicação

A solução disponibiliza o modelo treinado através de uma API RESTful utilizando:

- FastAPI
- Uvicorn
- Docker

---

## 📈 Monitoramento

A aplicação expõe métricas para observabilidade e monitoramento em produção utilizando:

- Prometheus FastAPI Instrumentator
- Logs estruturados
- Métricas HTTP
- Métricas de performance

---

# 🛠️ Tecnologias Utilizadas

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Deep Learning | TensorFlow / Keras |
| Redes Neurais | LSTM |
| Data Science | Pandas / NumPy |
| Pré-processamento | Scikit-Learn |
| Coleta de Dados | yfinance |
| API REST | FastAPI |
| Servidor ASGI | Uvicorn |
| Monitoramento | Prometheus |
| Containerização | Docker |

---

# 🧠 Arquitetura do Modelo

## Estrutura da Rede Neural

```text
LSTM(128)
   ↓
Dropout(0.2)
   ↓
LSTM(64)
   ↓
Dropout(0.2)
   ↓
Dense(25)
   ↓
Dense(1)
```

---

## ⚙️ Configurações Utilizadas

| Configuração | Valor |
|---|---|
| Tipo do Modelo | LSTM |
| Janela Temporal | 60 dias |
| Framework | TensorFlow / Keras |
| Otimizador | Adam |
| Função de Perda | Mean Squared Error |

---

# 📊 Resultados do Treinamento

O treinamento foi realizado localmente utilizando aceleração por GPU:

```text
NVIDIA RTX 3060
```

A utilização da GPU reduziu significativamente o tempo de treinamento e melhorou a convergência do modelo.

---

## 📈 Métricas Obtidas

| Métrica | Resultado |
|---|---|
| Épocas de Treino | 20 |
| Loss Final | 0.0015 |
| RMSE | 2.3 |
| MAE | 1.73 |

---

## 📌 Análise dos Resultados

Os resultados demonstram que o modelo conseguiu aprender padrões relevantes do comportamento financeiro das ações analisadas.

O erro médio absoluto (**MAE = 1.73**) indica que as previsões apresentam desvio médio aproximado de:

```text
± $1.73
```

Isso valida a eficácia do modelo para previsão de séries temporais financeiras.

---

# 📂 Estrutura do Repositório

```text
project/
│
├── train.py
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
├── scaler.pkl
└── lstm_stock_model.h5
```

---

# 📄 Descrição dos Arquivos

| Arquivo | Descrição |
|---|---|
| `train.py` | Coleta, treinamento e avaliação do modelo |
| `main.py` | API FastAPI com endpoints REST |
| `requirements.txt` | Dependências do projeto |
| `Dockerfile` | Configuração da imagem Docker |
| `scaler.pkl` | Scaler treinado |
| `lstm_stock_model.h5` | Modelo LSTM treinado |

---

# 🚀 Como Executar o Projeto

## 1️⃣ Criar Ambiente Virtual

### Windows

```bash
python -m venv venv
```

### Ativar Ambiente Virtual

```bash
.\venv\Scripts\activate
```

---

## 2️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

---

# 🧠 Treinamento do Modelo

Execute:

```bash
python train.py
```

Após o treinamento serão gerados:

- `lstm_stock_model.h5`
- `scaler.pkl`

---

# 🌐 Execução da API

Execute o servidor FastAPI:

```bash
uvicorn main:app --reload --port 9000
```

---

# 🔗 Acessos da Aplicação

| Serviço | URL |
|---|---|
| API | http://127.0.0.1:9000 |
| Swagger UI | http://127.0.0.1:9000/docs |
| Métricas | http://127.0.0.1:9000/metrics |

---

# 🐳 Execução com Docker

## Build da Imagem

```bash
docker build -t tech-challenge-fase4 .
```

---

## Executar Container

```bash
docker run -d -p 9000:9000 --name api_lstm tech-challenge-fase4
```

---

# 📊 Endpoints Disponíveis

| Endpoint | Método | Descrição |
|---|---|---|
| `/` | GET | Página inicial |
| `/predict` | POST | Realiza previsão financeira |
| `/metrics` | GET | Métricas Prometheus |
| `/docs` | GET | Documentação Swagger |

---

# 🌐 Exemplo de Requisição

## Endpoint

```http
POST /predict
```

---

## Payload

```json
{
  "prices": [
    101.2,
    102.5,
    103.1
  ]
}
```

> O array deve conter exatamente os últimos **60 preços de fechamento**.

---

# 📈 Monitoramento e Observabilidade

A API foi instrumentada utilizando:

```text
prometheus-fastapi-instrumentator
```

A solução fornece observabilidade em tempo real sobre:

- performance das requisições
- latência da API
- utilização de recursos
- comportamento do servidor
- estabilidade do sistema
- rastreamento de erros HTTP

---

## Endpoint de Métricas

```http
/metrics
```

Compatível com:

- Prometheus
- Grafana
- Datadog
- Loki
- Elastic Stack

---

# 📊 Principais Métricas Expostas

## ⏱️ Latência de Requisições

### Métrica

```text
http_request_duration_seconds
```

### Objetivo

Mede o tempo total necessário para processar cada requisição da API.

Permite:

- identificar gargalos
- medir tempo de inferência
- detectar lentidão
- acompanhar degradação de performance

---

## 🌐 Contagem de Requisições

### Métrica

```text
http_requests_total
```

### Objetivo

Realiza o rastreamento do volume de requisições recebidas.

Permite:

- monitoramento de tráfego
- análise de throughput
- identificação de picos de uso
- auditoria de consumo da API

### Endpoints Monitorados

- `/predict`

---

## 🧠 Utilização de Recursos e Memória

### Métrica

```text
python_gc_objects_collected_total
```

### Objetivo

Monitora a atividade do Garbage Collector do Python.

Fornece visão indireta sobre:

- uso de memória RAM
- reciclagem de objetos
- impacto do processamento de tensores
- comportamento durante inferência

### Benefícios

- detecção de memory leaks
- análise de estabilidade
- monitoramento sob alta carga
- suporte à otimização

---

## 📦 Tamanho das Respostas HTTP

### Métrica

```text
http_response_size_bytes
```

### Objetivo

Controla o volume de dados trafegados pela API.

Importante para:

- planejamento de escalabilidade
- otimização de banda
- análise de payloads
- redução de custos em cloud

---

# 📡 Telemetria Disponível

Ao acessar `/metrics`, o Prometheus recebe dados estruturados prontos para coleta automática.

---

## 🔍 Informações Expostas

| Informação | Valor |
|---|---|
| Python Version | 3.11.9 |
| Handlers Ativos | 4 |
| Status HTTP | 2xx / 4xx / 5xx |
| Framework | FastAPI |
| Servidor | Uvicorn |

---

# 📉 Observabilidade em Produção

O monitoramento implementado permite:

- 📌 acompanhamento em tempo real
- 📌 análise de disponibilidade
- 📌 detecção de falhas
- 📌 troubleshooting
- 📌 análise de comportamento da API
- 📌 monitoramento do modelo
- 📌 suporte à escalabilidade horizontal

---

# 📊 Integração com Grafana

As métricas expostas podem ser utilizadas para construção de dashboards avançados no Grafana.

Exemplos:

- Tempo médio de inferência
- Quantidade de previsões por minuto
- Uso de memória da aplicação
- Taxa de erros HTTP
- Latência média por endpoint
- Throughput da API
- Tempo de resposta P95/P99

---

# ⚙️ Escalabilidade e Arquitetura

A aplicação foi projetada considerando:

- Containerização completa
- Facilidade de deploy em cloud
- Separação entre treinamento e inferência
- Arquitetura orientada a serviços
- Compatibilidade com ambientes escaláveis

---

# 🚀 Benefícios da Arquitetura de Monitoramento

A estratégia de observabilidade implementada oferece:

- Baixo overhead
- Coleta automática
- Compatibilidade com ambientes cloud
- Facilidade de integração
- Monitoramento em tempo real
- Escalabilidade
- Maior confiabilidade operacional

---