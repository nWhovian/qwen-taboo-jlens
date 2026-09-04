# Qwen Taboo J-Lens: аудит готовности к минимальному write-up

Дата аудита: 2026-09-04. Статус отражает локальный repository и read-only
проверку текущего RunPod run. Он изменится после завершения notebook 07/08.

## Короткий диагноз

У проекта уже есть защищаемый каркас: exact revisions, BF16/FlashAttention 2,
published prompts, разделение validation/test, validation-frozen anchor,
одинаковые activations/positions для J-Lens и Logit Lens, leakage masks и
сохранение raw artifacts.

Но главный transfer claim пока **не заработан**. Не хватает:

1. законченного untouched 20-adapter test result;
2. base-model comparison на тех же prompts и targets;
3. output-prefix-only baseline, исключающего объяснение «секрет уже понятен из
   первых сгенерированных слов»;
4. чистого positive control для public J-Lens на base model;
5. paired uncertainty на уровне adapters/blocks и финальной human raw audit.

До этого допустима только формулировка: «pipeline работает, а validation даёт
обнадёживающий, но model-selected сигнал».

## Что уже есть

### Design and provenance

| Требование | Статус | Evidence / caveat |
|---|---:|---|
| Точный base model revision | ✅ | `Qwen/Qwen3.6-27B`, SHA `6a9e13b…` |
| Точные 20 adapter revisions | ✅ | Зафиксированы в test manifest |
| Точный J-Lens checkpoint/code | ✅ | `_n1000`, SHA `91271eb…`; official code `581d398…` |
| Primary runtime | ✅ | BF16, `flash_attention_2`, deterministic generation |
| Published prompt source and hash | ✅ | 100 standard + 100 direct test prompts |
| Adapter provenance as original-author artifacts | ⚠️ | Это публичные reproduction artifacts; original-author provenance не доказан |
| Exploration / validation / test separation | ✅ | Test не использован при выборе layer/position |
| Frozen confirmatory anchor | ✅ | Qwen layer 40, response offset 6 (`gen 5`) |
| Same cells for J-Lens and Logit Lens | ✅ | Общие residual activations, masks, targets |
| Raw artifact preservation | ✅ | Per-sequence Parquet/JSON and manifests |

### Validation evidence — не финальный результат

Run: `run_20260903T111007Z_qwen36_gold_blue_moon_validation`.

- 3 adapters × 30 standard validation prompts = 90 standard responses.
- 10 direct prompts per adapter were secondary; literal leaks excluded.
- Shared response-average layer 40 was selected by average performance across
  both methods, with a predeclared tie-break.
- At that validation-selected layer, top-5 response-level accuracy was 90/90
  (100%) for J-Lens and 86/90 (95.6%) for Logit Lens; both had Pass@10 and
  Majority@10 of 100%.
- Shared exact-position anchor was frozen at layer 40 / `gen 5`.
- Method-specific validation optima were stronger for J-Lens (layer 41,
  MRR 0.854, recall@1 0.811, recall@5 0.922) than for Logit Lens (layer 40,
  MRR 0.597, recall@1 0.544, recall@5 0.644), but these are selected validation
  numbers and must not be written as test evidence.

Interpretation: validation supports the choice of a common anchor and makes a
test worthwhile. It does not establish generalization, and response-average
top-5 is already near ceiling, so the confirmatory story should rely primarily
on exact-position rank/MRR and paired differences.

### Full test run — live snapshot

Run: `run_20260903T141427Z_qwen36_20_adapter_full_test`.

- Behavior generation is complete: 4,000 responses = 20 adapters ×
  (100 standard + 100 direct).
- Literal own-secret leaks: 15/2,000 standard (0.75%) and 135/2,000 direct
  (6.75%), 150 total. Leaks are excluded from headline readout metrics.
- At the final audit snapshot, 2,959/4,000 per-sequence position artifacts
  existed: 14 adapters complete and `salt` at 159/200.
- The persistent Jupyter kernel was still producing new files; do not restart
  or interrupt it.
- Notebook 08 test analysis had not yet been executed, so there are no valid
  confirmatory test numbers to place in the abstract.

Estimated remaining runtime at the observed recent rate: roughly 75–100 minutes
for readouts, then 20–45 minutes for aggregation/analysis if the kernel remains
healthy. This is an estimate, not a guarantee.

### Existing sanity checks

| Check | Статус | Assessment |
|---|---:|---|
| Tensor dimensions/layer mapping/numerics | ✅ | Passed |
| Emitted-token masking protocols | ✅ | Global mask primary; position-only/unmasked diagnostics saved |
| Target tokenization over 20 words | ✅ | Audited |
| Prompt literal target leakage | ✅ | Test prompt audit asserts none |
| Deterministic behavior smoke tests | ✅ | Base/correct/wrong tested on a small exploratory set |
| Human raw-output inspection | ⚠️ | A small sample was inspected; final stratified random audit still needed |
| Independent validation metric recomputation | ✅ | Layer-40 top-5 result reproduced from saved metric units |
| Independent test metric recomputation | ❌ | Test analysis not complete |
| Public J-Lens positive control on base | ⚠️ | Mars check verified plumbing, but J-Lens did not clearly beat LL; not a convincing semantic positive control |

## Главные альтернативные объяснения и необходимые тесты

### A. Это не transfer; target и так читается в base model

Почему возможно: current full test содержит correct-adapter conditions, но не
matched base-model condition на всех test prompts. Маленький Gold/Blue smoke
exploratory и недостаточен для systematic claim.

**Обязательный тест:** один base-model pass по тем же 200 test prompts без LoRA.
Для каждого prompt сохранить те же activations и оценить все 20 candidates на
layer 40 / `gen 5`; secondary — response-average layer 40. Сравнить correct LoRA
с base paired на одном prompt/target.

Что это различает:

- LoRA > base: поддерживает LoRA-specific target signal / transfer story.
- LoRA ≈ base: public lens может читать prompt/output semantics, но transfer
  через LoRA не показан.
- LoRA < base: compatible with degradation after adaptation.

### B. Секрет уже угадывается из видимого output prefix

Почему возможно: masking emitted token IDs удаляет прямое копирование токена из
readout vocabulary, но residual всё ещё кодирует семантику уже написанных слов.

**Обязательный простой baseline:** для каждого valid standard response дать
baseline только текст до точки `gen 5` и fixed list из 20 candidate secrets.
Измерить candidate rank/top-1/top-5. Желательно два варианта:

1. простой lexical/embedding retrieval;
2. blinded LLM multiple-choice judge с сохранёнными prompts/outputs и ручным
   аудитом случайных ошибок.

Full-output version полезен как upper bound, но matched prefix-only result
является главным control. Если judge load-bearing, показать минимум 20–30
случайных его решений и disagreement cases.

### C. Метод распознаёт generic Taboo/task state, а не собственный secret

**Обязательный analysis-only baseline:** использовать уже сохранённые scores
для всех 20 candidates.

- own target vs cyclic wrong target;
- shuffled target labels (permutation null);
- own-target probability share/margin against best wrong candidate;
- analytic chance: hit@1 = 0.05, hit@5 = 0.25, expected MRR for 20 random ranks
  ≈ 0.180.

Это не требует нового model run.

### D. J-Lens implementation работает технически, но не семантически

Текущий Mars smoke test недостаточен: он проверил pipeline, но не дал ясного
J-Lens advantage.

**Обязательный выбор перед claim “transfers”:**

- либо воспроизвести один маленький официальный base-model J-Lens example /
  benchmark на pinned checkpoint и показать ожидаемый сигнал;
- либо явно отказаться от сильного слова “transfers” и написать более узко:
  «a fixed base-fitted J-Lens decodes Taboo targets under these LoRAs», с
  caveat, что independent base semantic benchmark не был реплицирован.

Первый вариант предпочтительнее и должен быть дешёвым; не начинать новый fit.

### E. Pooled N завышает уверенность

Responses внутри одного adapter и paper block коррелированы. Wilson interval по
всем rows не отражает перенос между adapters.

**Обязательный анализ:** paired J-Lens − Logit Lens effect на каждом adapter;
cluster/bootstrap либо permutation inference с adapter как главным уровнем, а
prompt block как secondary resampling level. Показывать все 20 paired points и
интервал эффекта, а не только pooled mean.

## Что обязательно сделать до minimal write-up

### P0 — заработать один defensible headline result

1. **Дождаться текущего run.** Проверить 4,000 position + aggregate artifacts,
   final manifest/status, atomic Parquet readability и отсутствие пропусков.
2. **Запустить notebook 08 один раз без re-selection.** Сначала открыть только
   frozen standard test result: layer 40 / `gen 5`, global emitted-ID mask.
3. **Зафиксировать test result до exploratory scans.** MRR, median/geometric
   rank, hit@1/5/10, candidate share; JL−LL paired delta.
4. **Добавить uncertainty.** Adapter-level paired plot + cluster/bootstrap CI.
5. **Сделать base-model matched control.** Те же 200 prompts; primary analysis
   на standard non-leak examples.
6. **Сделать prefix-only output baseline.** Тот же information horizon `gen 5`.
7. **Посчитать wrong-target/shuffle null.** Из уже сохранённых 20-candidate
   scores.
8. **Сделать clean base J-Lens positive control** или сузить wording claim.
9. **Human audit.** Детерминированно выбрать минимум 30 standard и 30 direct
   examples, стратифицированных по adapters; проверить rendered prompt, target,
   output, leak flag, behavior relevance и необычные ranks.
10. **Независимо пересчитать headline metric** непосредственно из raw Parquet и
    сохранить короткий audit artifact.

### P1 — собрать минимальный package evidence

11. Standard prompts сделать primary; direct prompts — robustness/behavior
    diagnostic из-за более высокого leakage.
12. Response-average layer 40 оставить secondary paper-style comparison.
13. Mask sensitivity показать в appendix и проверить, меняется ли ordering
    методов.
14. All-layer/all-position test maps явно пометить exploratory; не выбирать по
    ним новый headline layer.
15. Выбрать случайные success/failure examples, а не только самые красивые.
16. Сохранить final tables, figure source data, configs, manifest, environment и
    exact artifact paths.

### Что не требуется для минимального claim

- Не нужен новый J-Lens fit, Oracle Lens, SAE или NLA.
- Не нужен causal intervention, если claim остаётся про decodability.
- Не нужно сравнивать абсолютные Qwen numbers с Gemma paper как matched result;
  это только context.
- Не нужны новые secrets или custom prompts до завершения текущего design.
- Не нужен второй generation seed при `do_sample=false`; uncertainty должна
  идти по prompts/adapters, а не по фиктивным seeds.

## Минимальный decision table после test

| Test outcome | Честный вывод |
|---|---|
| JL > LL, LoRA > base, prefix baseline слаб | Public base-fitted J-Lens transfers through these narrow LoRAs and adds target-specific decoding at the frozen anchor |
| JL > LL, но base ≈ LoRA | J-Lens decodes target-related semantics, но LoRA-specific transfer не установлен |
| JL ≈ LL | Public J-Lens transfers no better than the much simpler Logit Lens under this design |
| JL < LL | The base-fitted correction does not help, or degrades, after these LoRAs at the frozen anchor |
| Prefix-only baseline ≈ lenses | Signal is largely explained by visible output semantics; hidden-information claim fails |
| Эффект только post-hoc на других layers | Exploratory localization result; confirmatory advantage failed |
| Сильный эффект лишь у части adapters | Heterogeneous transfer; report adapter dependence rather than pooled generality |

## Реалистичный remaining-work estimate

При исправной текущей инфраструктуре:

- завершение текущего readout + notebook 08: **1.5–2.5 часа wall-clock**;
- base pass, positive control и output-prefix baseline: **2–4 часа**;
- raw audit, independent recomputation, paired statistics и figures:
  **1.5–2.5 часа**;
- заполнение executive summary и clean first draft: **1.5–2.5 часа**.

Итого до минимального честного draft с числами: примерно **6.5–11.5 часа**,
частично параллельно ожиданию GPU. Следующий checkpoint — завершённые 4,000/4,000
readout artifacts и frozen test table из notebook 08.
