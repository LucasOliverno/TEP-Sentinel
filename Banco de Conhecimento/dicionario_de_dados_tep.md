# Dicionário de Dados: Tennessee Eastman Process (TEP)

## Visão Geral
Este documento serve como referência rápida para **Operadores** e **Engenheiros**, detalhando as 52 variáveis monitoradas no painel de controle.

## 1. Medições de Processo (XMEAS)
Sensores instalados na planta que indicam *como* o processo está rodando.

### Alimentação (Feeds)
| Tag | Variável | Unidade | Explicação para o Operador | Valor Típico |
|:---:|---|:---:|---|:---:|
| **XMEAS 1** | A Feed (Stream 1) | kscmh | Vazão do Reagente A. Se cair, a reação para. | 0.25 |
| **XMEAS 2** | D Feed (Stream 2) | kg/h | Vazão do Reagente D. Principal matéria-prima pesada. | 3664 |
| **XMEAS 3** | E Feed (Stream 3) | kg/h | Vazão do Reagente E. Deve estar balanceada com D. | 4509 |
| **XMEAS 4** | A & C Feed (Stream 4) | kscmh | Vazão combinada de A e C (reciclados). | 9.35 |

### Unidade: Reator (Onde a mágica acontece)
| Tag | Variável | Unidade | Explicação para o Operador | Valor Típico |
|:---:|---|:---:|---|:---:|
| **XMEAS 6** | Reactor Feed Rate | kscmh | Total de gás entrando no reator. | 42.3 |
| **XMEAS 7** | **Reactor Pressure** | kPa g | **CRÍTICO**. Pressão interna. Se subir muito, risco de alívio/explosão. | 2705 |
| **XMEAS 8** | Reactor Level | % | Nível de líquido no reator. Deve cobrir os agitadores. | 75.0 |
| **XMEAS 9** | **Reactor Temp** | °C | **CRÍTICO**. Temperatura da reação. Controla a taxa de produção. | 120.4 |
| **XMEAS 21** | Reactor Cooling Temp | °C | Temperatura da água que sai da camisa de resfriamento. | 94.6 |

### Unidade: Separador (Separa Gás de Líquido)
| Tag | Variável | Unidade | Explicação para o Operador | Valor Típico |
|:---:|---|:---:|---|:---:|
| **XMEAS 11** | Separator Temp | °C | Temperatura dentro do vaso separador. | 80.1 |
| **XMEAS 12** | Separator Level | % | Nível de líquido acumulado no fundo. | 50.0 |
| **XMEAS 13** | Separator Pressure | kPa | Pressão no topo do separador. | 2634 |
| **XMEAS 14** | Separator Underflow | m3/h | Vazão de líquido sendo bombeada para o Stripper. | 25.2 |
| **XMEAS 22** | Sep Cooling Temp | °C | Temperatura da água do condensador do separador. | 77.3 |

### Unidade: Stripper (Purifica o Produto)
| Tag | Variável | Unidade | Explicação para o Operador | Valor Típico |
|:---:|---|:---:|---|:---:|
| **XMEAS 15** | Stripper Level | % | Nível de base da coluna. | 50.0 |
| **XMEAS 16** | Stripper Pressure | kPa | Pressão interna da coluna. | 3102 |
| **XMEAS 17** | Stripper Underflow | m3/h | **PRODUTO FINAL**. Vazão de produto saindo da planta. | 22.9 |
| **XMEAS 18** | Stripper Temp | °C | Temperatura da base. Garante remoção de impurezas. | 65.8 |
| **XMEAS 19** | Steam Flow | kg/h | Vazão de vapor injetado para aquecer a coluna. | 232 |

### Unidade: Compressor & Diversos
| Tag | Variável | Unidade | Explicação para o Operador | Valor Típico |
|:---:|---|:---:|---|:---:|
| **XMEAS 5** | Recycle Flow | kscmh | Gás não reagido que volta para o começo. | 26.9 |
| **XMEAS 10** | Purge Rate | kscmh | Gás jogado fora para evitar acúmulo de inertes. | 0.34 |
| **XMEAS 20** | Compressor Work | kW | Esforço do motor do compressor de reciclo. | 341.4 |

### Analisadores de Qualidade (Laboratório Online)
Estas variáveis indicam a composição (%) das correntes. O operador olha isso para ver a **Pureza**.

| Tag | Corrente Monitorada | Componentes | Foco |
|:---:|---|---|---|
| **XMEAS 23-28** | Reactor Feed | A, B, C, D, E, F | Balanceamento da reação. |
| **XMEAS 29-36** | Purge Gas | A, B, C, D, E, F, G, H | Perda de produto na purga. |
| **XMEAS 37-41** | **Product (Stripper Underflow)** | D, E, F, G, H | **QUALIDADE FINAL**. 'G' e 'H' são os produtos desejados. |

---

## 2. Variáveis Manipuladas (XMV)
Botões e Válvulas que o operador (ou o Agente de IA) pode mexer.
*Escala: 0% (Fechada) a 100% (Totalmente Aberta)*

| Tag | Nome Técnico | Explicação Ação | Efeito Principal |
|:---:|---|---|---|
| **XMV 1** | D Feed Valve | Abre alimentação de D | Aumenta pressão e nível do reator. |
| **XMV 2** | E Feed Valve | Abre alimentação de E | Aumenta pressão e nível do reator. |
| **XMV 3** | A Feed Valve | Abre alimentação de A | Aumenta pressão parcial de A. |
| **XMV 4** | A & C Feed Valve | Aumenta reciclo de A+C | Reposição de reagentes leves. |
| **XMV 5** | Compressor Recycle | Recirculação do Compressor | Aumenta vazão de reciclo; resfria reator. |
| **XMV 6** | **Purge Valve** | Abertura da Purga | Reduz pressão do sistema; joga massa fora. |
| **XMV 7** | Separator Liquid | Saída de líq. Separador | Controla Nível do Separador (XMEAS 12). |
| **XMV 8** | Stripper Liquid | Saída de líq. Stripper | Controla Nível do Stripper (XMEAS 15). |
| **XMV 9** | **Steam Valve** | Válvula de Vapor Stripper | Aquece o Stripper; remove impurezas do produto. |
| **XMV 10** | **Reactor Cooling** | Água Gelada do Reator | **SEGURANÇA**. Resfria o reator (baixa XMEAS 9). |
| **XMV 11** | Condenser Cooling | Água Gelada Condensador | Resfria saída do reator/entrada separador. |

---

## Observações de Segurança
1.  **Reação Exotérmica**: O reator gera calor. Se a válvula de água (**XMV 10**) falhar ou fechar, a temperatura (**XMEAS 9**) dispara, causando *runaway*.
2.  **Pressão**: Se a purga (**XMV 6**) ficar fechada, inerte acumula e a pressão (**XMEAS 7**) sobe até o alarme.
