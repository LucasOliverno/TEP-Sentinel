# Relatório de Verificação - Fase 3: Detecção de Falhas

## Objetivo
Validar se o Autoencoder treinado em dados normais consegue detectar anomalias (Falhas) sem gerar falsos alarmes excessivos.

## Metodologia
1.  **Modelo**: LSTM Autoencoder (Encoder-Decoder) treinado por 20 épocas.
2.  **Threshold**: Definido no percentil 99% do erro de reconstrução do treino.
3.  **Conjunto de Teste**:
    *   **Normal**: Dados nunca vistos pelo modelo (Teste de Falso Alarme).
    *   **Falha 1**: Alteração na razão A/C Feed (Teste de Detecção).

## Resultados Obtidos

| Métrica | Valor Obtido | Meta do Projeto | Status |
|---|---|---|---|
| **FAR (False Alarm Rate)** | **0.69%** | < 1% | ✅ Excelente |
| **MDR (Missed Detection Rate)** | **11.11%** | ~0-5% | ⚠️ Bom (89% TPR) |
| **TPR (True Positive Rate)** | **88.89%** | > 95% | ⚠️ Aceitável para v1 |

### Análise
*   O modelo é extremamente robusto a falsos positivos (apenas 0.7% de falsos alarmes), o que evita a "fadiga de alarme" nos operadores.
*   A taxa de detecção de ~89% para a Falha 1 mostra que o modelo aprendeu a dinâmica normal e percebe quando a planta sai do padrão.
*   Para melhorar o TPR, podemos testar janelas maiores ou arquiteturas mais profundas na Fase de Refinamento.

## Artefatos Gerados
*   **Modelo Final**: `models/tep_autoencoder.keras`
*   **Threshold**: `models/tep_threshold.pkl`
*   **Histograma de Erro**: `verification_output/verification_hist.png` (Visualização da separação entre Normal e Falha).
