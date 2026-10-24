# Pipeline de Processamento de Dados (Fase 2)

Código: `data_processor.py` · Dados: `python scripts/download_data.py` (Harvard Dataverse, Rieth et al. 2017).

## Decisões
1.  **Scaler**: `StandardScaler` ajustado **só** em `FaultFree_Training`, salvo em `models/tep_scaler.pkl` (versionado junto aos modelos).
2.  **Janelas** de 100 amostras, **por run**: nunca misturam `faultNumber`/`simulationRun`.
3.  **Rótulos**: 0 antes do início da falha (amostra 20 no treino, 160 no teste), `faultNumber` depois. O rótulo da janela é o do último passo.
4.  **Split por `simulationRun`**: `run % 5 == 0` é validação. Janelas sobrepostas do mesmo run ficam no mesmo lado.
5.  **Memória**: leitura falha por falha (filtros do pyarrow) e saída em `.npy` pré-alocado, lido com `mmap`.

## Saída (`processed_data/{train,test}/`)
`X.npy` (N,100,52 float32), `y.npy`, `run.npy`, `fault.npy`, `meta.json`.

Com os strides padrão do código: treino 20 → 220.500 janelas; teste 50 → 189.000 janelas.

```python
from data_processor import load_split, run_split_masks
X, y, run, fault = load_split("train")      # X é memmap
train_mask, val_mask = run_split_masks(run)
```
