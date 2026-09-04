# Практический гайд по research write-up в стиле Neel Nanda

Это рабочая компиляция требований из текущего MATS application doc, публичных
постов Нила о research process и writing, его paper-writing checklist и
примеров прошлых заявок, которые в application doc помечены как принятые.

Здесь важно различать:

- **явное требование** — прямо сказано в application doc или checklist;
- **устойчивый паттерн** — повторяется в принятых примерах и комментариях Нила;
- **рекомендация для нашего проекта** — вывод, а не цитата Нила.

## 0. Формальные требования текущего MATS application task

- Исследовательская часть рассчитана примерно на **16 часов** и не должна
  превышать **20 часов**; ещё около **2 часов** можно потратить на executive
  summary и application form.
- Основной deliverable — Google Doc, начинающийся с executive summary; в нём
  нужны многочисленные понятные graphs и достаточно деталей, чтобы проследить
  работу без чтения кода.
- Executive summary: ориентир около одной страницы, допустимо 1–3 страницы,
  верхняя граница — примерно 600 слов. Графики можно и нужно включать.
- В summary должны быть: problem/motivation, high-level takeaways и по одному
  короткому paragraph + graph на каждый ключевой эксперимент.
- Если данные или LLM judge являются load-bearing, сразу после summary нужно
  показать случайные качественные примеры и ошибки.
- Короткий ответ в application form читается первым как фильтр. Он должен
  содержать конкретику: модели, главный эксперимент, ключевое число,
  surprising finding и biggest limitation.
- Нельзя сдавать сырой LLM-generated prose. LLM можно использовать для critique,
  outline и drafting, но автор обязан понимать и проверить каждую формулировку.

## 1. Что оценивается

### Clarity

- [ ] Cold-start читатель понимает вопрос без знания проекта и кода.
- [ ] В первых абзацах названы модель, данные, вмешательство/метод и главная
      метрика.
- [ ] Каждое утверждение можно связать с конкретным экспериментом и рисунком.
- [ ] Из текста ясно, что именно показал эксперимент и чего он не показал.
- [ ] Executive summary самодостаточен.

### Detail and reproducibility

- [ ] Описаны генерация данных, выбор prompts/splits и критерии исключения.
- [ ] Указаны модели, revisions, precision, attention backend, seeds и
      generation parameters.
- [ ] Метрики определены, включая единицу анализа и способ агрегации.
- [ ] Есть достаточно деталей, чтобы понять эксперимент без чтения кода.
- [ ] Код, README и удобный notebook позволяют воспроизвести headline result.

### Research taste

- [ ] Вопрос интересен и связан с alignment / model understanding.
- [ ] Есть точная развилка между несколькими правдоподобными гипотезами.
- [ ] Сначала сделана простая и дешёвая проверка существования феномена.
- [ ] Сложность каждого следующего метода оправдана тем, что не может показать
      более простой эксперимент.
- [ ] Scope позволяет глубоко проверить 1–2 результата.

### Skepticism and truth-seeking

- [ ] Выписана сильнейшая простая альтернативная гипотеза.
- [ ] Для неё есть дешёвый, решающий control или baseline.
- [ ] Проверены leakage, metric mismatch, cherry-picking и grader gaming.
- [ ] Просмотрены случайные сырые примеры, а не только лучшие.
- [ ] Хотя бы одна headline metric пересчитана независимо от основного анализа.
- [ ] Отрицательный или смешанный результат допускается заранее.
- [ ] Ограничения не спрятаны и не смягчены маркетинговым языком.

### Technical depth, practicality, prioritisation

- [ ] Есть рабочий end-to-end pipeline и сохранённые raw artifacts.
- [ ] Главные решения следуют из research question, а не из доступности красивых
      графиков.
- [ ] На ключевой вывод приходится несколько независимых линий evidence, если
      claim систематический.
- [ ] Неудачные направления и pivots упомянуты только там, где они объясняют
      выбор финального эксперимента.
- [ ] Write-up организован логически, а не как дневник по времени.

## 2. Research process: Explore → Understand → Distill

Нил описывает четыре стадии: ideation, exploration, understanding и
distillation. Их нельзя оценивать одной и той же меркой.

### Exploration: максимизировать information gain

- Быстро набирать surface area: простые эксперименты, визуализации, сырые
  примеры, аномалии.
- Не требовать от себя окончательной гипотезы слишком рано.
- Регулярно спрашивать: «сколько новой информации я получаю за единицу
  времени?»
- Вести highlights doc с наиболее странными и полезными результатами.
- Результаты exploration нельзя задним числом выдавать за confirmatory test.

### Understanding: различать гипотезы

- Записать гипотезы и предсказания до решающего теста.
- Сделать эксперимент, результаты которого вероятнее при одной гипотезе, чем
  при другой.
- Искать не только подтверждение любимой гипотезы, но и объяснения вида «это
  просто leakage / prompt copying / generic topic detection».
- Количественные выводы делать на заранее выбранной или held-out выборке.
- Qualitative evidence брать из случайной выборки; cherry-picked примеры
  допустимы только как иллюстрации после систематического результата.

### Distillation: сжать исследование до нескольких истин

- Сначала выписать 1–3 максимально узких claims.
- Для каждого claim собрать минимальный достаточный набор evidence.
- Red-team каждый claim как скептический reviewer.
- Если при написании обнаружилась дыра, вернуться к understanding; это нормальный
  результат письма, а не провал.
- Писать, чтобы информировать, а не чтобы убедить любой ценой.

Публичные источники: [Explore, Understand, Distill](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand),
[Key Mindsets](https://www.alignmentforum.org/posts/cbBwwm4jW6AZctymL/my-research-process-key-mindsets-truth-seeking),
[Research Taste](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research).

## 3. Как спроектировать хороший эксперимент

### До запуска

- [ ] Сформулировать один главный вопрос в форме сравнения или развилки.
- [ ] Зафиксировать возможные исходы, включая null/negative result, и их
      интерпретацию.
- [ ] Назвать экспериментальную единицу: prompt, response, block, adapter,
      seed или model checkpoint.
- [ ] Отделить exploration, validation/model selection и untouched test.
- [ ] Зафиксировать primary metric, layer/position, mask и exclusion rules до
      теста.
- [ ] Зафиксировать stopping/pivot rule.
- [ ] Проверить, что claim соответствует данным: existence claim требует
      меньшего evidence bar, чем «метод систематически лучше baseline».

### Baselines и controls

- [ ] **Naive chance baseline:** что получится при случайном угадывании?
- [ ] **Strong simple baseline:** самый простой метод, который может объяснить
      результат.
- [ ] **Negative control:** условие, где эффект не должен появляться.
- [ ] **Positive control:** условие, где pipeline обязан найти известный сигнал.
- [ ] **Matched control:** меняется только исследуемый фактор.
- [ ] **Random / shuffled control:** random vector, shuffled labels/data или
      matched wrong target, если это применимо.
- [ ] **Input/output-only baseline:** можно ли получить ответ из наблюдаемого
      текста без внутренних activations?
- [ ] Все методы сравниваются на одинаковых examples, layers, positions,
      masks и target definitions.
- [ ] Baseline получает честный model-selection budget; нельзя тюнить новый
      метод сильнее, чем baseline.

### Данные и измерение

- [ ] Проверить сырые prompts, rendered prompts, token IDs и outputs.
- [ ] Проверить отсутствие target в input и определить literal/semantic output
      leakage.
- [ ] Аудировать tokenization каждого target и словоформы.
- [ ] Сохранить raw records до aggregation.
- [ ] Не считать коррелированные responses независимыми наблюдениями.
- [ ] Для paired методов использовать paired difference и uncertainty на
      уровне настоящей независимой единицы, а не только pooled rows.
- [ ] Число примеров и exclusions показывать как абсолютные числа и доли.
- [ ] Предпочитать effect size + confidence interval одному p-value.

### Agent/LLM sanity checks — особенно важный блок

- [ ] Человек просмотрел случайную выборку prompts, outputs и transcripts.
- [ ] Load-bearing claim сопоставлен с raw artifacts и кодом.
- [ ] Headline metric независимо пересчитана другим коротким путём.
- [ ] Критический эксперимент повторён или проверен альтернативной
      реализацией, где это возможно.
- [ ] Проверено не менее ~75% остальных экспериментов на ошибки и
      несогласованности.
- [ ] LLM judge не является непрозрачной точкой отказа; его prompts, outputs и
      случайные ошибки показаны читателю.
- [ ] В write-up явно описано, какие проверки были сделаны человеком.

## 4. Минимальная структура MATS write-up

### Executive summary: примерно одна страница, максимум 3 страницы / 600 слов

Рекомендуемый порядок:

1. **Problem and motivation.** Один абзац: что неизвестно и почему это важно.
2. **Setup.** Модель, данные, сравниваемые методы и held-out design.
3. **Takeaway 1.** Одно точное утверждение, число и главный график.
4. **Takeaway 2.** Второе утверждение, число и график — только если оно
   действительно отдельное и защищённое.
5. **Largest limitation.** Что мешает более сильной интерпретации.

Executive summary и короткое form summary читаются первыми и работают как
фильтр. В них нужны конкретные model names, sample sizes, surprising numbers и
limitations, а не общие слова.

### Основной текст

1. **Research question and claims** — 1–2 claims, сформулированные уже с
   ограничениями.
2. **Background** — только знания, необходимые для чтения результатов.
3. **Experimental design** — splits, models, conditions, metrics, pre-registered
   choices, exclusions.
4. **Result 1** — зачем эксперимент нужен, какие исходы ожидались, figure,
   result, interpretation, alternative explanation.
5. **Result 2** — та же структура; не добавлять слабый результат для объёма.
6. **Sanity checks and baselines** — отдельный видимый раздел.
7. **Qualitative examples** — случайные примеры и failures.
8. **Limitations** — прямой список того, чего работа не устанавливает.
9. **Related work** — только связи, реально меняющие интерпретацию.
10. **Conclusion** — повторить узкий итог без новых claims.
11. **Appendix** — hyperparameters, полные таблицы, дополнительные plots,
    prompt templates, репликационные детали.

### Abstract для более формальной версии

Context/problem → точный contribution/claim → главные числа/evidence →
implication. После чтения abstract нельзя задаваться вопросом «что конкретно
они сделали и что получили?»

Полное руководство: [Highly Opinionated Advice on How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers).
Checklist Нила: [My Checklist for Writing ML Conference Papers](https://docs.google.com/document/d/1AoF6bPJp-muWnsZLMmfcxo1fmAu1izUzZXDFHar-35o/edit?usp=sharing).

## 5. Как строить narrative

- Начинать не с хронологии, а с 1–2 вещей, которые теперь можно считать более
  вероятными.
- Для каждого result объяснять, зачем эксперимент был нужен и какие исходы
  различали гипотезы.
- Самый сильный честный claim лучше широкого впечатляющего claim.
- Если простой baseline закрывает преимущество метода, это и есть результат.
- Failure/pivot полезен, когда показывает judgement: заметили проблему,
  изменили эксперимент и получили более информативный ответ.
- Не называть readout «what the model thought»: это decodability данным методом
  на конкретном layer/position, не causal use.
- Не скрывать post-hoc analyses: явно помечать exploratory.

### Шаблон одного result section

> **Question.** Какую альтернативу мы проверяем?
>
> **Design.** Что фиксировано, что меняется, какая выборка и метрика?
>
> **Prediction.** Что ожидать при H1 и H0?
>
> **Result.** Одно главное число с uncertainty и figure.
>
> **Interpretation.** Какое узкое обновление следует?
>
> **Caveat/control.** Какое простое объяснение ещё остаётся или уже исключено?

## 6. Figures

- [ ] До построения figure записан один intended takeaway.
- [ ] Figure связан с core claim, а не просто красив.
- [ ] Caption читается самостоятельно: setup, N, metric, direction, exclusions,
      uncertainty.
- [ ] На графике отмечены validation-frozen choices и exploratory regions.
- [ ] Есть raw/paired points или distribution, а не только среднее.
- [ ] Оси, units и legends читаемы без увеличения.
- [ ] Не использовать красно-зелёную пару как единственный канал различия.
- [ ] Небольшое количество важных элементов визуально выделено.
- [ ] В main text 2–4 figures; полный grid и sensitivity plots — в appendix.

## 7. Что показывают принятые примеры

Эти выводы основаны на текущем application doc Нила и связанных там работах,
а не означают, что каждый пример является идеальным paper.

### R1D1

[Write-up](https://docs.google.com/document/d/1OiqmJ36EgBgzy5sR4YEFrn4WFXEQFmZ0oc5t8579ysE/edit?tab=t.0#heading=h.ljkfgfirrgkt)

- Короткий (около 1.4k слов), понятный executive summary.
- Сначала поставленный вопрос, затем неудачный initial approach и разумный
  inverse/pivot.
- Явные random/original/reasoning controls и ограничения.
- Главный урок: ясность, прагматичный pivot и новый конкретный insight могут
  быть сильнее большого количества результатов.

### Empathic Machines

[Write-up](https://crawling-opossum-1a2.notion.site/Empathic-machines-1a44cd7fb1b780539302c6c50a5ca80c)

- Executive summary начинается с двух research questions.
- Метод и результаты даны компактными блоками, с figures и raw examples.
- Автор прямо называет toy synthetic dataset большим ограничением.
- Главный урок: крупное ограничение не дисквалифицирует работу, если evidence
  убедителен внутри узкой области применимости.

### What Impacts CoT Faithfulness

[Write-up](https://docs.google.com/document/d/11U0Mg2boJSCp8GVc15mhvxW0Kg23b6my8NN5oAv4vM0/edit?tab=t.0)

- Точные models/sample sizes и почти «один finding — один graph».
- Есть neutral control и скептическая проверка.
- Комментарий Нила: интересный и хорошо приоритизированный проект, но текст
  местами трудно читать и он предполагает слишком много контекста.
- Главный урок: хорошие результаты не компенсируют cold-start clarity.

### “Wait” / backtracking

[Write-up](https://docs.google.com/document/d/1wX5rpAXc5VrOxZ9hvfTd_1g3UifXTzrUUg8ie3SQQqk/edit?tab=t.0#heading=h.mfdxfkjn7xkt)

- RQ1/RQ2, точные counts, несколько методов, failures и limitations.
- Очень продуктивно и технически сильно.
- Комментарий Нила: scope слишком широк, отдельным findings не хватает глубины.
- Главный урок: не превращать продуктивность в пять конкурирующих narratives.

### R1 Distill Diffing

[Write-up](https://docs.google.com/document/d/1_-zmL_8xm-jypTqei0yU7NwrpFv2H-loiwRGxwpn6l4/edit?tab=t.0)

- Честно показывает, что основной метод не сработал, и извлекает информацию из
  dataset/pattern analysis.
- Демонстрирует прагматизм, но в определении model-specific latent была
  концептуальная ошибка.
- Главный урок: pivot ценен; центральное определение нужно проверять особенно
  тщательно.

### SAE Equations

В application doc этот пример принят за хороший вкус, мотивацию и сильные
qualitative results, несмотря на меньшую продуктивность. Урок: одна качественная
линия evidence может быть достаточной, если claim соответственно узкий.

Каталог и оценки: [Neel Nanda MATS 12.0 Admissions Procedure + FAQ](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.tf1kwj1t9c19).

## 8. Common failure modes: финальный red-team checklist

- [ ] Мы не принимаем успешный plot за доказательство без raw inspection.
- [ ] Нет простой альтернативы, которую можно было проверить за час, но мы её
      не проверили.
- [ ] Phenomenon существует именно на этой модели, revision и prompt format.
- [ ] Baseline действительно сильный и не искусственно ослаблен.
- [ ] Test не использовался для layer/position/metric selection.
- [ ] Exclusions зафиксированы до просмотра test result.
- [ ] Qualitative examples не cherry-picked.
- [ ] Нет псевдорепликации в uncertainty.
- [ ] Claim не шире model family, adapters, prompts и positions эксперимента.
- [ ] Декодируемость не названа причинностью.
- [ ] Отрицательные результаты и biggest limitation видны в executive summary.
- [ ] В тексте нет сырого LLM prose: автор проверил каждое предложение,
      определение и число.

## 9. Definition of done для минимального честного write-up

- [ ] Есть один untouched confirmatory result.
- [ ] Для него есть сильный простой baseline, positive control и negative/matched
      control.
- [ ] Есть paired effect size с корректной uncertainty.
- [ ] Просмотрены случайные raw examples и показаны типичные failures.
- [ ] Headline metric независимо пересчитана.
- [ ] Есть 1-page executive summary с 1–2 claims, числами, figures и biggest
      limitation.
- [ ] Все post-hoc scans помечены exploratory.
- [ ] Reader может найти configs, raw data, analysis notebook и exact revisions.
- [ ] Автор способен устно объяснить, почему каждый главный graph меняет мнение
      о конкретной гипотезе.
