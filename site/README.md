# J-Lens x Taboo: all-adapter prompt browser

Static GitHub Pages site generated from saved run `run_20260903T141427Z_qwen36_20_adapter_full_test`. It covers all
20 saved Taboo adapters, with `smile` selected
by default. For every adapter, exact duplicate answers are collapsed within each
prompt type and the browser keeps 50 lexically diverse examples. It starts from
25 direct plus 25 standard and fills any uniqueness shortfall from the other
prompt type.

The compact published data include layer-by-position target ranks, probability
masses, and top-1 decoded tokens for J-Lens and Logit Lens. Distinct literal
own-secret leaks are retained and visible. Sample selection does not use lens
outcomes. No model weights, credentials, or hidden activations are published.
