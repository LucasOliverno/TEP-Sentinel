# Pipeline de Processamento de Dados (Fase 2)

## Visão Geral
Este documento descreve como os dados brutos (Parquet) foram transformados em Tensores prontos para o treinamento de Redes Neurais (CNN-LSTM/Autoencoders).

## Estratégia Adotada

### 1. Prevenção de Data Leakage
*   **Scaler**: `StandardScaler` (média 0, desvio padrão 1).
*   **Fit**: O scaler foi treinado **EXCLUSIVAMENTE** nos dados `FaultFree_Training`.
    *   Isso garante que o modelo não conheça a estatística das falhas ou do conjunto de teste antecipadamente.
*   **Persistência**: Salvo em `processed_data/tep_scaler.pkl`.

### 2. Janelamento (Time Series)
Os dados industriais possuem inércia. Uma amostra isolada ($t$) não contem informação suficiente. Usamos janelas deslizantes (Sliding Windows).

| Configuração | Treino (Train) | Teste (Test) | Motivo |
|---|---|---|---|
| **Janela ($T$)** | 100 amostras | 100 amostras | Captura ~3 mins de histórico. |
| **Stride (Passo)** | 10 | 100 | **Train**: Stride pequeno cria overlap (Data Augmentation) para mais dados.<br>**Test**: Stride = Janela (Sem overlap) para avaliação justa e economia de memória. |

### 3. Tensores Gerados
Os arquivos finais em `processed_data/` são arquivos comprimidos Numpy (`.npz`):

*   **tep_train.npz**:
    *   **X**: `(524,991, 100, 52)` -> 525k amostras de treino.
    *   **y**: Etiquetas de falha (0 = Normal, 1-20 = Falhas).
*   **tep_test.npz**:
    *   **X**: `(100,800, 100, 52)` -> 100k amostras de teste.
    *   **y**: Etiquetas correspondentes.

## Como carregar?
```python
import numpy as np
data = np.load('processed_data/tep_train.npz')
X_train = data['X']
y_train = data['y']
```

## Próximos Passos (Fase 3)
*   Treinar Autoencoder para detecção de anomalias (usando apenas dados normais do `tep_train.npz` onde `y=0`).
