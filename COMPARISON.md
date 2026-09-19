# Comparison

Measured on one RTX 3060. Rows are identical across systems where a
number was measured here; published numbers are marked as such.

## OpenJev fixtures

| system | authored144 | WANLI256 | TypeSafe102 agreement | TypeSafe102 variation |
|---|---:|---:|---:|---:|
| Jev (TypeSafe, closed) | — | — | 0.8831 | 0.1268 |
| OpenJev, frozen Qwen3.5-4B | 0.8132 | 0.6365 | 0.8453 | 0.177 |
| typedec v10-doc | 0.7131 | 0.5744 | 0.4582 | 0.4915 |
| typedec v11-tools | — | — | — | — |
| typedec v6-base | 0.7194 | 0.6293 | 0.4094 | 0.4929 |
| typedec v9-route | 0.7369 | 0.5707 | 0.3305 | 0.4977 |

## General test set

| stratum | v10-doc | v11-tools | v6-base | v9-route |
|---|---|---|---|---|
| anli-r1-test | 0.5575 | 0.5650 | 0.5575 | 0.5900 |
| anli-r2-test | 0.3925 | 0.3825 | 0.4025 | 0.3775 |
| arc-easy-test | 0.5775 | 0.5975 | 0.5825 | 0.5675 |
| boolq-val | 0.7850 | 0.7850 | 0.8050 | 0.7900 |
| clinc-test | 0.9350 | 0.9475 | 0.9375 | 0.9425 |
| mmlu-test | 0.3075 | 0.3025 | 0.2925 | 0.3000 |
| openbookqa-test | 0.4200 | 0.4175 | 0.4175 | 0.4075 |
| snli-test | 0.7150 | 0.7275 | 0.7350 | 0.7400 |
| sst5-test | 0.4975 | 0.4600 | 0.4800 | 0.4675 |
| **mean** | 0.5701 | 0.5749 | 0.5737 | 0.5713 |

## Question banks

| stratum | v10-doc | v11-tools | v6-base | v9-route |
|---|---|---|---|---|
| belebele-de | 0.6300 | 0.6200 | 0.6425 | 0.6500 |
| belebele-en | 0.8400 | 0.8225 | 0.8375 | 0.8300 |
| medmcqa-test | 0.3175 | 0.2975 | 0.3075 | 0.2850 |
| sciq-test | 0.9875 | 0.9925 | 0.9900 | 0.9850 |
| **mean** | 0.6924 | 0.6821 | 0.6915 | 0.6852 |

## Ten languages

| stratum | v10-doc | v11-tools | v6-base | v9-route |
|---|---|---|---|---|
| belebele-ar | 0.2950 | 0.3000 | 0.2950 | 0.2450 |
| belebele-de | 0.5350 | 0.5550 | 0.5650 | 0.6000 |
| belebele-en | 0.8100 | 0.7950 | 0.8150 | 0.8100 |
| belebele-es | 0.5800 | 0.5800 | 0.6200 | 0.5900 |
| belebele-fr | 0.6000 | 0.5800 | 0.5950 | 0.5950 |
| belebele-hi | 0.2750 | 0.3300 | 0.3000 | 0.3300 |
| belebele-it | 0.5650 | 0.5500 | 0.5600 | 0.5600 |
| belebele-pt | 0.4900 | 0.5250 | 0.5000 | 0.5250 |
| belebele-ru | 0.5000 | 0.4700 | 0.4900 | 0.5150 |
| belebele-zh | 0.5750 | 0.5750 | 0.5950 | 0.5500 |
| xnli-test-ar | 0.4650 | 0.4650 | 0.4750 | 0.4650 |
| xnli-test-de | 0.6100 | 0.6450 | 0.6250 | 0.6500 |
| xnli-test-en | 0.8600 | 0.8550 | 0.8550 | 0.8250 |
| xnli-test-es | 0.7000 | 0.7050 | 0.6900 | 0.6800 |
| xnli-test-hi | 0.4550 | 0.4550 | 0.4800 | 0.4350 |
| xnli-test-ru | 0.5550 | 0.5700 | 0.5950 | 0.5800 |
| xnli-test-tr | 0.4850 | 0.4550 | 0.4750 | 0.4750 |
| xnli-test-vi | 0.5300 | 0.5900 | 0.5800 | 0.5300 |
| xnli-test-zh | 0.6450 | 0.6450 | 0.6550 | 0.6450 |
| **mean** | 0.5551 | 0.5593 | 0.5669 | 0.5591 |

## Tool selection

| stratum | v11-tools |
|---|---|
| tools-glaive | 0.9971 |
| tools-hermes | 0.9828 |
| **mean** | 0.9928 |

## Browser actions

| stratum | v11-tools |
|---|---|
| browser-mind2web | 0.6550 |
| **mean** | 0.6769 |

## Routing economics

Quality is the share of routed items the chosen tier answers.
Spend is relative, one unit is the small tier.

| strategy | tier accuracy | quality | relative spend |
|---|---:|---:|---:|
| typedec | 0.4733 | 0.4793 | 1.08 |
| always_small | 0.4633 | 0.4633 | 1.00 |
| always_middle | 0.4520 | 0.8987 | 4.00 |
| always_frontier | 0.0560 | 0.9227 | 90.00 |
| random | 0.4633 | 0.4633 | 1.00 |
| oracle | 1.0000 | 1.0000 | 7.31 |
