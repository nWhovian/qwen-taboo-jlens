# Qwen Taboo J-Lens: что уже готово и чего не хватает

Дата: 4 сентября 2026. Этот список относится к текущему короткому write-up, а
не к попытке сделать полноценную статью.

## Короткий диагноз

Минимальный честный write-up уже можно писать. Основной claim заработан полным
TEST на 20 адаптерах и двумя сильными controls:

- public base J-Lens остаётся информативной после narrow LoRA;
- сигнал специфичен к правильному адаптеру, а не только к prompt или generic
  Taboo state;
- средний результат J-Lens лучше Logit Lens, но улучшение неоднородно и не
  является universal win.

Rock refit, общий quality-control и J-space можно включить как короткие
exploratory follow-ups. Они делают сюжет интереснее, но не должны становиться
главным claim.

## Что завершено

| Проверка | Статус | Главный результат |
|---|---:|---|
| 20-adapter TEST, standard + direct | ✅ | 4,000/4,000 responses и readouts |
| Literal leak exclusion | ✅ | standard 15/2,000; direct 135/2,000 |
| J-Lens vs Logit Lens, одинаковые cells/masks | ✅ | standard response-average Hit@5 79.6% vs 69.9% |
| Validation-frozen early anchor | ✅ | `layer 40 / gen 5`: 67.6% vs 51.2% |
| Matched base-model control | ✅ | Hit@5 0% у обеих линз |
| True secret vs 19 wrong secrets | ✅ | correct top-1 96.0% J-Lens, 94.7% LL |
| Adapter-level heterogeneity | ✅ | J-Lens MRR выше для 11/20; CI эффекта пересекает ноль |
| Raw-output audit | ✅ agent-side | Нашлись visible hints, off-task outputs и missed obfuscated leaks |
| Independent headline recomputation | ✅ | Hit@1/5 и 20-way accuracy совпали |
| Rock-specific J-Lens refit + evaluation | ✅ exploratory | public lens выше по MRR; одинаковый R@5 на `gen 5` |
| 2 × 2 general-quality control | ✅ exploratory | общего MRR gap почти нет; Rock-specific excess gap есть |
| Public J-space on Rock | ✅ exploratory | exact `rock` 2/99, но `rocks` top-1 87/99 |

## Что обязательно сделать до отправки минимального write-up

Это уже не новые GPU experiments, а короткая проверка текста и evidence:

1. **Автору просмотреть deterministic qualitative sample.** Особенно примеры
   `moon`, `rock`, два off-task ответа и obfuscated leaks. Не оставлять выбор
   красивых примеров полностью агенту.
2. **Сверить каждую цифру с `EVIDENCE_MAP.md`.** Не переносить в summary
   post-hoc layer scans или validation numbers как TEST evidence.
3. **Оставить claim про decodability.** Не писать “the model thought”,
   “stored knowledge” или “causally used”.
4. **Назвать J-space metric post-hoc.** `rock` → `rocks` — содержательная
   находка, но morphology-aware scoring появилось после просмотра raw support.
5. **Проверить provenance wording.** Это public Qwen reproduction adapters, а
   не обязательно checkpoints оригинальных авторов Taboo.
6. **Оставить две основные фигуры.** Base/wrong-secret specificity и
   per-adapter heterogeneity. Остальное достаточно описать текстом.

После этих шагов write-up уже минимально готов. Не нужно ждать ещё одного
большого запуска.

## Самые важные недостающие эксперименты

### P0, если хотим усилить refit-механизм

**Matched base `n=100` lens.** Обучить base-model J-Lens на тех же 100 neutral
sequences, на которых обучена Rock-specific lens. Затем сравнить public
`n=1000`, base `n=100` и Rock `n=100` на одних и тех же 99 Rock rows.

Сейчас general-quality control показывает, что Rock-lens не просто глобально
хуже, но ещё не разделяет sample-count, corpus и model-condition confounds.
Без matched `n=100` нельзя уверенно объяснять, почему public lens лучше читает
секрет.

### P0, если хотим claim «информация beyond visible output»

**Output-prefix-only baseline at `gen 5`.** Дать baseline только текст ответа,
видимый к этой позиции, и 20 candidate secrets. Измерить correct top-1 и rank.

Текущий token mask убирает буквальное копирование, но не смысл подсказок. Пока
этого baseline нет, нельзя утверждать, что white-box readout находит больше,
чем можно понять из текста.

### P0, если хотим заявить улучшение от J-space

**Predefined morphology-aware replication.** До нового просмотра outputs
задать правила lexical family для нескольких secrets и повторить decomposition.
Нужно включить разные типы слов, особенно адаптеры, где ordinary J-Lens слабее.

Текущий Rock результат позволяет написать “J-space selected a nearby plural
form”, но не “J-space improves recovery”.

## Полезно, но не обязательно для этого write-up

- Taboo adapters, специально обученные сопротивляться prompt attacks.
- Multi-token secrets или более реалистичная hidden objective.
- Fresh model family и другой public lens checkpoint.
- Causal swap/ablation только для отдельного causal claim.
- Full-output black-box baseline как upper bound.

## Финальная формулировка результата

Хорошая версия:

> A public base-model J-Lens remained informative across 20 narrow Taboo LoRAs
> and decoded the correct adapter secret far above matched base and wrong-secret
> controls. Its average advantage over Logit Lens was useful but heterogeneous.

Слишком сильные версии, которых данные пока не поддерживают:

- “J-Lens always outperforms Logit Lens.”
- “The lens reveals what the model is thinking.”
- “The public lens recovers information unavailable from the output.”
- “The Rock-specific lens follows censorship while the public lens bypasses
  it.”
- “J-space improves recovery.”

Последние две фразы остаются хорошими hypotheses for next work, а не готовыми
выводами.
