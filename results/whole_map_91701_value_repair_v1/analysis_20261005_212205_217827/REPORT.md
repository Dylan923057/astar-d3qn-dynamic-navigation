# Value repair probe audit

Source: `D:\Asatr-D3QN-Workshop\results\whole_map_91701_value_repair_v1\probe_20261005_211357_827670`

Every recorded checkpoint uses the same 50 fixed validation scenes, epsilon=0, and network-only actions. No test data.

| Method | Seed | Safe | Dynamic collision | Timeout | Static goal |
|---|---:|---:|---:|---:|---:|
| unguided_bound | 0 | 0% | 2% | 98% | False |
| advice_raw | 0 | 0% | 6% | 94% | False |

Failure waiting, repeated movement, and periodic-tail flags can overlap. Remaining static BFS distance is recorded for every failure.

A 2000-step probe checks integration and early numerical behavior. It cannot establish convergence, final superiority, or a publication claim.
