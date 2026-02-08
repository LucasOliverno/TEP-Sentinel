# Relatório Técnico Final: TEP-Sentinel

**Data**: 07/02/2026
**Versão**: 1.0 (Sistema Completo)

---

## 1. Visão Geral do Sistema

O **TEP-Sentinel** é um sistema de controle e monitoramento inteligente para o Tennessee Eastman Process (TEP). Ele combina técnicas de Deep Learning (Detecção de Falhas), Reinforcement Learning (Controle Autônomo) e Generative AI (Diagnóstico Explicável) em uma plataforma unificada.

### Arquitetura de Alto Nível

O sistema opera em um ciclo fechado (Closed-Loop) gerenciado pelo Orquestrador (`tep_system.py`):

1.  **Sensoriamento**: O ambiente (`TEPEnv`) gera o estado atual (52 variáveis).
2.  **Monitoramento (FDD)**: O "Cérebro" analisa os dados em busca de anomalias.
3.  **Controle (RL)**: O Agente toma decisões para estabilizar o processo.
4.  **Explicação (RAG)**: Se uma falha é detectada, o módulo LLM consulta manuais técnicos e explica a causa.
    > **RAG (Retrieval-Augmented Generation)**: Técnica que combina o poder de geração de texto de LLMs (como GPT ou Gemini) com a precisão de uma base de conhecimento externa, reduzindo alucinações.
5.  **Interface**: Tudo é visualizado em tempo real no Dashboard (`dashboard.py`).

---

## 2. Detalhamento dos Módulos

### Fase 1 & 2: Engenharia de Dados
*   **Scripts**: `convert_data.py`, `data_processor.py`
*   **Fluxo**:
    *   Conversão de dados brutos (`.RData`) para `parquet`.
        > **Nota Técnica**: *Parquet* é um formato de armazenamento colunar otimizado para Big Data, permitindo leitura muito mais rápida e eficiente que CSVs tradicionais.

    *   **Normalização**: `StandardScaler` treinado apenas em dados normais (evita vazamento de dados de falha).
    *   **Janelamento**: Criação de sequências temporais ($T=100$) para alimentar as redes neurais.

### Fase 3: Detecção e Diagnóstico (FDD)
*   **Detecção (Autoencoder)**:
    *   *Arquivo*: `models/tep_autoencoder.keras`
        > **Nota Técnica**: `.keras` é o formato nativo do TensorFlow para salvar modelos completos (arquitetura + pesos + otimizador) em um único arquivo compactado.
    *   *Lógica*: Reconstrói a janela de entrada. Se o Erro Quadrático Médio (MSE) > Threshold, marca como **FALHA**.
*   **Diagnóstico (Classificador)**:
    *   *Arquivo*: `models/tep_classifier.keras`
    *   *Lógica*: Rede CNN-LSTM que recebe a janela de falha e classifica qual o tipo (IDV 1 a 20).

### Fase 4: Controle Inteligente (RL)
*   **Ambiente**: `envs/tep_env.py`
    *   Utiliza um *Surrogate Model* (LSTM) para simular a física do TEP.
        > **Conceito**: *LSTM (Long Short-Term Memory)* é uma arquitetura de rede neural recorrente capaz de aprender dependências de longo prazo em séries temporais, ideal para modelar a dinâmica lenta de processos químicos.
    *   Possui **Camada de Segurança** (Safety Layer) que penaliza ações perigosas.
*   **Agente**: PPO (Proximal Policy Optimization).
    > **Conceito**: *PPO* é um algoritmo de Reinforcement Learning de última geração (estado da arte) que equilibra a exploração do ambiente com a estabilidade do aprendizado, evitando mudanças drásticas na política de controle.
    *   *Arquivo*: `models/PPO/best_model.zip`
    *   *Objetivo*: Manter as variáveis de processo (XMEAS) próximas de zero (Setpoints).

### Fase 5: IA Explicável (RAG)
*   **Banco Vetorial**: `chroma_db/` (ChromaDB).
*   **Indexador**: `rag_indexer.py`.
    *   Processou manuais técnicos do diretório `Banco de Conhecimento/`.
    *   Usa embeddings do Google Gemini (`models/embedding-001`).
*   **Agente RAG**: `rag_agent.py`.
    *   Quando o FDD detecta uma falha, este agente:
        1.  Recebe o Código da Falha (ex: IDV(1)).
        2.  Busca documentos relevantes no ChromaDB.
        3.  Envia para o **Gemini 2.0 Flash** gerar um relatório em linguagem natural.

### Fase 6: Integração e UI
*   **Orquestrador**: `tep_system.py`.
    *   Classe `TEPSentinelSystem`.
    *   Mantém os modelos carregados em memória e gerencia o buffer de dados (Rolling Window).
*   **Dashboard**: `dashboard.py`.
    *   Desenvolvido em Streamlit.
    *   Permite controle da simulação (Start/Stop/Reset).
    *   Possui **Injeção de Falhas** para testes de robustez.

---

## 3. Guia de Arquivos (Project Structure)

```text
TEP-Sentinel/
├── Banco de Conhecimento/      # Documentação e Manuais (Fonte do RAG)
├── envs/
│   └── tep_env.py              # Ambiente Gym (Simulador)
├── models/                     # Pesos treinados (.keras, .zip, .pkl)
├── processed_data/             # Scalers e dados processados
├── tep_system.py               # <--- CÉREBRO CENTRAL (Orquestrador)
├── dashboard.py                # <--- INTERFACE (Streamlit)
├── rag_agent.py                # Módulo de Consulta ao LLM
├── rag_indexer.py              # Script de Indexação de documentos
├── data_processor.py           # Pipeline de Engenharia de Features
├── train_rl.py                 # Script de Treinamento do Agente
├── train_autoencoder.py        # Script de Treinamento do FDD
└── requirements.txt            # Dependências
```

## 4. Como Executar

Para iniciar o sistema completo:

### Método Recomendado (Windows)

Dê um clique duplo no arquivo `run_dashboard.bat` na pasta do projeto.

### Método Manual (Terminal)

```bash
run_dashboard.bat
```
_Ou execute diretamente com o Python configurado:_
```bash
C:/Users/soare/AppData/Local/Programs/Python/Python312/python.exe -m streamlit run dashboard.py
```

Isso iniciará o Dashboard no navegador, onde você pode interagir com todo o ecossistema TEP-Sentinel.
