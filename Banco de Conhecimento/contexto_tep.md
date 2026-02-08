# O Contexto do Tennessee Eastman Process (TEP)

## O que é o TEP?
O **Tennessee Eastman Process (TEP)** é um modelo matemático de simulação de uma planta química real, criado e disponibilizado pela empresa Eastman Chemical Company em 1993 (Downs & Vogel).

Desde então, ele se tornou o **"Padrão Ouro" (Benchmark)** mundial para testar novas tecnologias de:
*   Controle de Processos (PID, MPC, etc.)
*   Detecção de Falhas (FDD - Fault Detection and Diagnosis)
*   Segurança Operacional

### Por que ele é tão famoso?
Antes do TEP, as pesquisas acadêmicas usavam modelos muito simples (tanques de nível, fornos pequenos). O TEP trouxe a complexidade da vida real para o computador:
1.  **Altamente Não-Linear**: As reações químicas mudam drasticamente com temperatura e pressão.
2.  **Instável (Open-Loop Unstable)**: Se você desligar os controladores, o reator "explode" (matematicamente) em menos de 1 hora.
3.  **Multivariável**: Tudo afeta tudo. Mexer na válvula do separador altera a pressão do reator lá atrás.

---

## Como funciona a "Fábrica Virtual"?

O processo produz dois produtos químicos líquidos (**G** e **H**) a partir de quatro reagentes gasosos (**A, C, D, E**) e um inerte (**B**).

### Fluxo Simplificado (Para Entendimento Geral)

1.  **Alimentação**: Os gases A, C, D e E entram no sistema.
2.  **Reator (O Coração)**:
    *   Aqui ocorre a reação química exotérmica (gera calor).
    *   Transforma gases em produtos líquidos.
    *   *Perigo*: Requer resfriamento constante (água gelada) para não superaquecer.
3.  **Condensador**:
    *   Resfria a saída do reator para transformar o vapor em líquido.
4.  **Separador (O Filtro)**:
    *   Separa o que virou líquido (vai para frente) do que ainda é gás (volta para o reator/compressor).
5.  **Compressor de Reciclo**:
    *   Pega o gás que não reagiu e manda de volta para o reator (economia de matéria-prima).
6.  **Stripper (A Refinaria Final)**:
    *   Remove as últimas impurezas dos produtos G e H usando vapor.
    *   O que sai no fundo do Stripper é o nosso **Produto Vendável**.

---

## O Desafio TEP-Sentinel

O TEP-Sentinel não é apenas mais um controlador. Ele é uma camada de inteligência acima do controle básico.

### O Problema
A simulação padrão do TEP inclui **20 Falhas Programadas** que desafiam a operação:
*   Falha 1: Mudança na composição da alimentação A/C.
*   Falha 4: Temperatura da água de resfriamento sobe.
*   Falha 6: Perda de alimentação A (vazamento na linha).
*   Falha Desconhecida (21+): O modelo permite criar novas anomalias.

### Nossa Missão
Usar Inteligência Artificial para:
1.  **Detectar** essas falhas antes que o alarme toque (Fase 3).
2.  **Diagnosticar** qual é a causa raiz (Fase 3/5).
3.  **Agir** (via Controle RL) para estabilizar a planta automaticamente (Fase 4).

> **Resumo**: O TEP é o nosso "simulador de voo". Se nossa IA conseguir pilotar essa planta química instável e cheia de falhas, ela pode pilotar qualquer processo industrial real.
