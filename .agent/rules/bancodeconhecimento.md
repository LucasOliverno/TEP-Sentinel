# Diretrizes do Banco de Conhecimento

## Propósito
Este documento define a regra de documentação contínua do projeto. O objetivo é transformar o processo de desenvolvimento em uma base de conhecimento educativa, permitindo que qualquer agente (humano ou IA) entenda não apenas *o que* o código faz, mas *por que* ele foi construído dessa forma.

## Regra Áurea
**Toda vez que uma nova funcionalidade for implementada, um conceito complexo for aplicado ou uma decisão arquitetural for tomada, a IA DEVE criar ou atualizar um arquivo explicativo na pasta `/Banco de Conhecimento`.**

## Estrutura da Documentação

Os arquivos devem ser salvos na pasta raiz: `TEP-Sentinel/Banco de Conhecimento/`.

### Tipos de Conteúdo Esperados:

1.  **Tutoriais "Hands-On"**:
    *   Explicam como o código funciona passo a passo.
    *   Exemplo: `entendendo_o_loader_de_dados.md` (Explica como o arquivo `.RData` é convertido).

2.  **Conceitos Teóricos (Deep Learning/Controle)**:
    *   Explicam a teoria por trás da implementação.
    *   Exemplo: `teoria_autoencoder_para_anomalias.md` (Explica o porquê de usar erro de reconstrução).

3.  **Decisões de Projeto (ADR - Architecture Decision Records)**:
    *   Registram o motivo de escolher X ao invés de Y.
    *   Exemplo: `escolha_chromadb_vs_pinecone.md`.

## Formato do Arquivo

Use Markdown com uma linguagem didática e clara.

```markdown
# [Título do Conceito/Funcionalidade]

## O que é?
Explicação simples e direta.

## Por que estamos usando no TEP-Sentinel?
Contexto específico do projeto.

## Como foi implementado?
Trechos de código com comentários explicativos.
Link para o arquivo fonte: `[arquivo.py](../caminho/arquivo.py)`

## Próximos Passos
O que muda a partir daqui?
```

## Exemplo de Fluxo
1. IA implementa a normalização dos dados.
2. IA cria `Banco de Conhecimento/normalizacao_e_janelamento.md`.
3. O arquivo explica que usamos `StandardScaler` treinado só no dataset "Fault-Free" para não contaminar os dados com informações das falhas (Data Leakage).
