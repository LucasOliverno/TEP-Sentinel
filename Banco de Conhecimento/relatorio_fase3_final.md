# Relatório Final - Fase 3: Detecção e Diagnóstico

Reproduzir: `python scripts/download_data.py && python data_processor.py && python train_autoencoder.py && python calibrate_threshold.py && python train_classifier.py && python evaluate.py`

## Visão geral
1.  **Detector de anomalias (Autoencoder LSTM)**: treinado só com dados normais; alarme quando o erro de reconstrução da janela passa do limiar, por 3 amostras seguidas.
2.  **Diagnosticador (CNN-LSTM)**: classifica a janela em 21 classes (0 = normal/desconhecida, 1-20 = IDV). As probabilidades são suavizadas (média das últimas 10 janelas) durante o alarme.

## Metodologia (corrigida em relação à v1)
*   Janelas **nunca cruzam** fronteiras de `(faultNumber, simulationRun)`.
*   Amostras anteriores ao início da falha (amostra 20 no treino, 160 no teste) são rotuladas **0**. Na v1 o run inteiro recebia o rótulo da falha, o que limitava artificialmente o recall a ~86% e a precisão da classe 0 a 30%.
*   Treino/validação separados **por `simulationRun`** (janelas com 90% de overlap do mesmo run não ficam dos dois lados).
*   Limiar calibrado em runs normais retidas, no nível de **episódios de alarme** com a mesma lógica de debounce do sistema (`calibrate_threshold.py`), e não por percentil de janelas.
*   Avaliação por run no conjunto de **teste** (`evaluate.py`), com atraso de detecção.

## Resultados (conjunto de teste, 10 runs por falha)
Threshold=0.8573 | stride=1 | confirm=3 | runs/fault=10 | diagnosis read 50 samples after the alarm

- False alarms, fault-free runs: 0.27% of windows, 10.00% of runs raise a confirmed alarm
- False alarms, pre-onset part of faulty runs: 1.80% of windows, 10.00% of runs

| IDV | Description | Detection rate | Window TPR | Median delay (samples) | Diagnosis acc. (alarmed runs) |
|---|---|---|---|---|---|
| 1 | A/C feed ratio, B composition constant (stream 4 | 100% | 99% | 8 | 100% (10) |
| 2 | B composition, A/C ratio constant (stream 4) - s | 100% | 98% | 16 | 100% (10) |
| 3 | D feed temperature (stream 2) - step change (pra | 10% | 0% | 707 | 100% (1) |
| 4 | Reactor cooling water inlet temperature - step c | 100% | 99% | 7 | 100% (10) |
| 5 | Condenser cooling water inlet temperature - step | 100% | 57% | 6 | 100% (10) |
| 6 | A feed loss (stream 1) - step change | 100% | 100% | 3 | 100% (10) |
| 7 | C header pressure loss, reduced availability (st | 100% | 100% | 1 | 100% (10) |
| 8 | A, B, C feed composition (stream 4) - random var | 100% | 96% | 28 | 90% (10) |
| 9 | D feed temperature (stream 2) - random variation | 30% | 0% | 694 | 100% (3) |
| 10 | C feed temperature (stream 4) - random variation | 100% | 73% | 91 | 100% (10) |
| 11 | Reactor cooling water inlet temperature - random | 100% | 98% | 14 | 100% (10) |
| 12 | Condenser cooling water inlet temperature - rand | 100% | 99% | 9 | 100% (10) |
| 13 | Reaction kinetics - slow drift | 100% | 94% | 52 | 100% (10) |
| 14 | Reactor cooling water valve - sticking | 100% | 100% | 3 | 100% (10) |
| 15 | Condenser cooling water valve - sticking (practi | 20% | 1% | 429 | 100% (2) |
| 16 | Unknown | 100% | 66% | 78 | 100% (10) |
| 17 | Unknown | 100% | 95% | 38 | 100% (10) |
| 18 | Unknown | 100% | 92% | 61 | 100% (10) |
| 19 | Unknown | 100% | 92% | 52 | 100% (10) |
| 20 | Unknown | 100% | 94% | 54 | 100% (10) |

### Leitura honesta
*   **IDV 3, 9 e 15 são indetectáveis** (detecção ≈ taxa de falso alarme; é um resultado conhecido da literatura, essas falhas quase não alteram as medições). A acurácia de diagnóstico dessas linhas vem de 1-3 alarmes e não tem significado.
*   As demais 17 falhas são detectadas em 100% das runs. O atraso mediano vai de 1 a ~90 amostras (3 min/amostra).
*   **Falsos alarmes**: ~10% das runs normais de 960 amostras levantam um alarme confirmado (≈ 1 a cada 10 mil amostras). É um compromisso: um limiar mais alto reduz isso e piora o recall das falhas sutis (10, 16).
*   O diagnóstico é lido 50 janelas após o alarme. Logo no instante do alarme a janela ainda contém quase só dados normais, então o classificador só estabiliza depois de algumas dezenas de amostras.
*   O MSE por janela fica ~0,75 em operação normal porque o ruído de medição do TEP não é reconstruível. O sinal útil é o aumento sobre esse piso.
