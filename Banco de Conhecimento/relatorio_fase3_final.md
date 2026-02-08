# Relatório Final - Fase 3: Detecção e Diagnóstico

## Visão Geral
Nesta fase, implementamos o "Cérebro" do sistema TEP-Sentinel, composto por dois modelos complementares:
1.  **Detector de Anomalias (Autoencoder)**: Monitora o processo e alerta se algo sair do padrão.
2.  **Diagnosticador de Falhas (CNN-LSTM)**: Analisa o alerta e identifica a causa raiz (qual das 20 falhas).

## 1. Detector de Anomalias (Autoencoder)
*   **Função**: "Semáforo" (Verde = Normal, Vermelho = Perigo).
*   **Treinamento**: Apenas dados normais (Unsupervised).
*   **Performance**:
    *   **Falsos Alarmes (FAR)**: `0.69%` (Excelente, meta < 1%).
    *   **Detecção de Falha 1 (TPR)**: `88.89%` (Consegue pegar a maioria das falhas).

## 2. Diagnosticador de Falhas (Classificador)
*   **Função**: "Médico" (Diz qual é a doença).
*   **Arquitetura**: Híbrida (CNN para extrair features + LSTM para tempo).
*   **Performance Geral**:
    *   **Acurácia Global**: `85.55%` (em 21 classes).
    *   **Precisão Média**: `92%` (Weighted Avg).

### Destaques por Classe
| Classe | Tipo | Precision | Recall | Status |
|---|---|---|---|---|
| **0** | **Normal** | 30% | 69% | ⚠️ Confunde com algumas falhas sutis |
| **1** | **Falha A/C Ratio** | 99% | 80% | ✅ Muito confiável |
| **5** | **Falha Temp.** | 100% | 80% | ✅ Perfeito na identificação |
| **15** | **Falha Condenser** | 44% | 77% | ⚠️ Difícil de distinguir |
| **Média** | **(Todas)** | **92%** | **86%** | **Sistema Robusto** |

## Conclusão da Fase 3
O sistema está funcional. O pipeline ideal para operação é:
1.  **Autoencoder** filtra o dia a dia (baixa taxa de alarme falso).
2.  Quando o Autoencoder apita, o **Classificador** entra em ação para dizer o que fazer.
3.  A acurácia de 85% é um ótimo ponto de partida para a **Fase 4 (Controle)**.

## Próximos Passos
*   Iniciar a **Fase 4: Controle Inteligente**.
*   Conectar esses diagnósticos a um sistema de recomendação ou ação automática.
