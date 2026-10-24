# Relatório - Fase 3: Autoencoder

Os números da v1 (FAR 0,69%, TPR 88,89%) foram removidos: vinham de um script não versionado, de um limiar calibrado no próprio treino e de rótulos que consideravam a falha ativa desde a amostra 0.

A avaliação atual está em `relatorio_fase3_final.md` (gerada por `evaluate.py`, em `verification_output/evaluation.md`).

## Modelo
*   LSTM(64) → LSTM(16, gargalo) → RepeatVector → LSTM(16) → LSTM(64) → Dense(52).
*   Treino: janelas normais de runs `simulationRun % 5 != 0`; validação nas runs `% 5 == 0`. Seed fixa (42).
*   Limiar: `calibrate_threshold.py`, menor valor que mantém os episódios de falso alarme dentro de um orçamento nas runs normais retidas.

## Artefatos
*   `models/tep_autoencoder.keras`, `models/tep_threshold.pkl`, `models/tep_scaler.pkl`.
