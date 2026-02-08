# Ingestão de Dados: Conversão de .RData para Parquet

## O que é?
Este processo consiste em ler os arquivos de dados originais da simulação do Tennessee Eastman Process (TEP), que estão no formato `.RData` (nativo do R), e convertê-los para o formato `.parquet`, que é um formato de armazenamento colunar altamente eficiente e amplamente suportado em Python (Pandas/Polars).

## Por que estamos usando no TEP-Sentinel?
Os dados originais do TEP (Rieth et al., 2017) são distribuídos como arquivos `.RData`. No entanto, nosso pipeline de Processamento e Deep Learning será construído em Python. 
- O formato **Parquet** foi escolhido em vez de CSV porque:
    1.  **Compressão**: Reduz drasticamente o tamanho em disco (arquivos de ~800MB RData podem virar centenas de MB em Parquet, enquanto em CSV poderiam passar de 2GB).
    2.  **Tipagem Preservada**: Os tipos de dados (float64, int, etc.) são mantidos sem necessidade de redefinição na leitura.
    3.  **Velocidade**: A leitura e escrita de Parquet é ordens de grandeza mais rápida que CSV no Pandas.

## Como foi implementado?
Utilizamos a biblioteca `pyreadr` para a leitura dos arquivos R e `pyarrow` como motor para salvar em Parquet.

### Script de Conversão (`convert_data.py`):
```python
import pyreadr
import pandas as pd

# Exemplo de fluxo para um arquivo
result = pyreadr.read_r("TEP_FaultFree_Training.RData")
df = result['fault_free_training']
df.to_parquet("TEP_FaultFree_Training.parquet", index=False)
```

## Estrutura dos Dados Ingeridos
Cada DataFrame contém 52 variáveis do processo:
- **XMEAS (1 a 41)**: Variáveis de medição (Temperaturas, Pressões, Composições).
- **XMV (1 a 11)**: Variáveis manipuladas (Posições de válvulas, fluxos).
- **faultNumber**: Identificador da falha (0 = Normal, 1-20 = Falhas específicas).
- **simulationRun**: ID da rodada de simulação.
- **sample**: Índice temporal da amostra.

## Próximos Passos
Agora que temos os arquivos em Parquet, iniciaremos a **Análise Exploratória (EDA)** para entender a distribuição das variáveis e a separabilidade das falhas.
