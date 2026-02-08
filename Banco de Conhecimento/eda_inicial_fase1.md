# Relatório de Análise Exploratória (EDA) - Fase 1

## O que foi feito?
Realizamos uma análise estatística e visual dos dados do TEP (Rieth et al., 2017) para validar sua integridade e entender o comportamento do processo.

## Principais Descobertas

### 1. Correlação (Redundância)
A matriz de correlação do estado Normal revelou fortes acoplamentos, esperados em processos químicos:
*   **Pressão e Temperatura do Reator**: Altamente correlacionados. Se um sobe, o outro tende a subir (lei dos gases + cinética).
*   **Vazão e Nível**: Em malhas de controle fechadas, o nível dita a vazão de saída.

### 2. Assinatura de Falhas
Comparando o estado Normal vs. Falha 1 (Alteração na Composição de A/C Feed):

| Variável | Desvio (Drift) | Significado Físico |
|---|---|---|
| **XMEAS 19 (Strip Steam)** | +51.1 | O controlador tenta compensar impurezas aumentando o vapor no Stripper. |
| **XMV 3 (A Feed Valve)** | +47.9 | A válvula de alimentação A abre drasticamente para tentar manter a estequiometria. |
| **XMEAS 7 (Reactor Press)** | +6.4 | A pressão do reator sobe consideravelmente. |

### 3. Dinâmica Temporal
Os gráficos de série temporal (salvos em `eda_output/`) mostram que o processo não salta instantaneamente para o novo estado. Há um **período transiente** (inércia térmica e hidráulica) antes de estabilizar ou instabilizar.
*   Isso confirma a necessidade de usar **Janelas Temporais (LSTMs)** na Fase 2. Modelos estáticos falhariam em capturar essa dinâmica.

## Conclusão para o Projeto
Os dados são coerentes com a física do problema. As falhas deixam "digitais" claras (drifts) nas variáveis, o que valida a viabilidade de treinar modelos de detecção (Fase 3).

## Próximos Passos
Ir para a **Fase 2: Engenharia de Atributos**:
1.  Normalizar os dados (usando apenas a estatística do estado Normal para evitar vazamento).
2.  Criar as janelas temporais (Tensors).
