# FDD evaluation (held-out test set)

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
