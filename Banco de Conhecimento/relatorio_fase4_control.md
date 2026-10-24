# Relatório - Fase 4: Controle Inteligente (RL)

## Escopo e limitações
Não há simulador Fortran do TEP no projeto. O ambiente `TEPEnv` usa um **modelo substituto neural** (LSTM) da dinâmica da planta, treinado só em operação normal.

*   O modelo aprende **correlações**: nos dados, as XMV são a saída dos PIDs da própria planta, então o efeito causal de uma válvula não é identificável por completo.
*   O RL aqui é uma **demonstração de rejeição de perturbação em torno do estado nominal**, dentro do modelo substituto. **Não está validado contra a planta real** nem contra as falhas.
*   O sistema de detecção/diagnóstico (Fase 3) roda sobre runs reais do dataset. O substituto só alimenta o modo "Surrogate" do painel.

## Ambiente (`envs/tep_env.py`)
*   Cada episódio começa com as 41 XMEAS perturbadas (N(0, 1) em unidades padronizadas). O agente precisa levá-las de volta a 0.
*   O substituto recebe um contexto de 20 passos (o LSTM precisa de histórico; na v1 recebia 1 passo).
*   Recompensa: `-mean(XMEAS²) - 0,01·mean(ação²)`. Passar de 10σ = planta "explodiu" (−100, episódio termina).
*   Estado retornado por cópia (sem aliasing) e `reset(seed)` reprodutível.

## Modelo substituto
*   MSE em runs retidas: **0.418**, contra **0.990** do baseline de persistência (x[t+1] = x[t]). A v1 reportava loss ~19, pior que prever a média.

## Agente PPO (`train_rl.py`, 100 mil passos, seed 42)
*   baseline_zero_action: recompensa média -503.1, XMEAS MSE final 3.22, explosões 0
*   ppo_best: recompensa média -105.6, XMEAS MSE final 0.62, explosões 0

(20 episódios com seeds fixas; baseline = nenhuma ação.) O melhor checkpoint por avaliação vai para `models/PPO/best_model.zip`, que o sistema carrega.
