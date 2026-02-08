---
trigger: always_on
---

# Contexto do Projeto: TEP-SENTINEL

## Visão Geral
O projeto TEP-SENTINEL visa desenvolver um sistema de diagnóstico e controle inteligente para processos industriais, focado no Tennessee Eastman Process (TEP). O pipeline vai desde a aquisição de dados brutos até a tomada de decisão autônoma e assistida por LLMs (RAG), garantindo estabilidade, segurança e eficiência operacional.

## Roadmap Detalhado

### Fase 1: Fundação e Ingestão de Dados
*   **Foco**: Garantir a integridade dos dados brutos.
*   **Atividades Chave**:
    *   Aquisição dos dados da simulação (baseado em Rieth et al., 2017).
    *   Conversão de formatos proprietários (`.RData`) para DataFrames manipuláveis.
    *   Análise Exploratória (EDA) para entender correlações entre as 52 variáveis do processo.

### Fase 2: Engenharia de Atributos (Feature Engineering)
*   **Foco**: Preparar dados para modelos deep learning (temporais).
*   **Atividades Chave**:
    *   **Normalização**: Uso de StandardScaler treinado *apenas* em dados normais (Fault-free) para preservar anomalias.
    *   **Janelamento**: Criação de tensores 3D (amostras x time_steps x features) para capturar dinâmica temporal.
    *   **Rotulagem**: Mapeamento de classes (Estado Normal + 20 tipos de falhas).

### Fase 3: Desenvolvimento do Cérebro (FDD)
*   **Foco**: Detecção e diagnóstico de falhas.
*   **Modelos**:
    *   **Unsupervised**: Autoencoders para detecção de anomalias (monitoramento de erro de reconstrução).
    *   **Supervised**: Arquiteturas CNN-LSTM para classificação específica das falhas.
*   **Refinamento**: Otimização de hiperparâmetros (learning rate, dropout) para robustez contra ruído.

### Fase 4: Controle e Reinforcement Learning (RL)
*   **Foco**: Estabilização autônoma.
*   **Atividades Chave**:
    *   Integração com ambiente `pc-gym`.
    *   Definição de Espaço de Ação: 11 variáveis manipuladas (XMVs).
    *   Reward Shaping: Penalizar variabilidade excessiva e violações de limites de segurança.
    *   **Safety Layer**: Implementação de restrições lógicas irremovíveis para segurança crítica.

### Fase 5: Módulo LLM (RAG)
*   **Foco**: Explicação e suporte à decisão (IA Explicável).
*   **Arquitetura**:
    *   **Knowledge Base**: Manuais técnicos e históricos do TEP.
    *   **Vector DB**: Chroma ou Pinecone.
    *   **RAG Pipeline**: Integração de alertas do FDD com contexto técnico para gerar relatórios em linguagem natural.
    *   **Output**: Recomendações acionáveis para operadores.

### Fase 6: Métricas de Negócio (KPIs)
*   **Metas**:
    *   **FAR (False Alarm Rate)**: < 1% (evitar fadiga de alarme).
    *   **MDR (Missed Detection Rate)**: ~ 0% para falhas críticas.
    *   Estabilidade do processo (redução de variabilidade).

## Diretrizes para Agentes de IA

1.  **Tecnicidade**: Mantenha o rigor técnico ao manipular dados industriais. Erros de unidade ou escala são críticos.
2.  **Segurança em Primeiro Lugar**: Ao sugerir código de controle (Fase 4), nunca ignore as camadas de segurança (Safety Layers).
3.  **Contexto Temporal**: Lembre-se que os dados são séries temporais; validação cruzada deve respeitar a ordem temporal, não aleatória.
4.  **Modularidade**: Mantenha o código de ingestão, treinamento e inferência desacoplados para facilitar a manutenção.
5.  **Documentação**: Todo código crítico deve ser documentado, especialmente as lógicas de recompensa em RL e arquiteturas de redes neurais.
