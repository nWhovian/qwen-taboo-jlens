# Can a Base-Model J-Lens Decode a Taboo Word After LoRA Fine-Tuning?

## Short answer

Yes, in this experiment it can.

We tested a public J-Lens on 20 versions of Qwen3.6-27B. Each version had a
small LoRA adapter. Each adapter was trained around one secret word, such as
`moon`, `rock`, or `smile`. The model had to give hints about the word without
saying it.

At the fifth generated token, the public J-Lens put the correct word family in
its top 5 in **74.0%** of leak-free answers. Logit Lens reached **52.3%**. A
sparse J-Space readout reached **68.0%**.

The public J-Lens was trained on the base model, before any of these LoRAs were
added. So the main result is simple:

> A J-Lens trained on the base model can still read a strong and specific
> signal after narrow LoRA fine-tuning.

## Why we did this

Interpretability tools are often trained for one exact model. But real models
are fine-tuned many times. It would be useful if one tool could still work on a
related fine-tuned model.

This project tests that question in a small and controlled setting. It is not a
competition to find the best way to guess a Taboo word. Black-box prompting can
work very well here. We wanted to know whether a reusable **white-box** tool can
still read the internal signal after LoRA fine-tuning. Here, white-box means
that the method can see the model's internal activations.

## What we tested

We used `Qwen/Qwen3.6-27B`, 20 public Taboo LoRAs, and 100 standard test prompts
per adapter. We compared a public J-Lens trained on base Qwen, Logit Lens, and
public J-Space with 16 sparse components.

Logit Lens reads an activation directly as token scores. J-Lens first adjusts
it with a map learned from the model. J-Space tries to describe it with a small
set of J-Lens token directions.

We chose layer 40 and the fifth generated token on a separate validation set.
We did this before looking at the full 20-adapter test result.

The full test had 2,000 answers. We removed 17 that contained the secret or its
simple plural form. This left **1,983 answers per method**.

We also masked token IDs that appeared in the model's answer. This stops a
method from getting credit for directly copying an emitted token. It does not
remove the meaning of the words already written in the answer.

A hit counts either the singular or a simple plural form: for example, `rock`
or `rocks`. We added this rule after the Rock pilot showed that morphology
matters, then fixed it for all 20 adapters. This is still a follow-up, not a
completely untouched confirmation.

## Main result

The table shows how often the correct word family appeared among the decoded
words at layer 40 and generated token 5.

| Method | Top 1 | Top 5 | Top 10 |
|---|---:|---:|---:|
| Logit Lens | 34.3% | 52.3% | 60.0% |
| Public J-Lens | **55.7%** | **74.0%** | **78.2%** |
| Public J-Space | 53.9% | 68.0% | 70.6% |

![Morphology-aware recovery for all 20 adapters.](../reports/all_adapter_jspace_gen5/run_20260904T162622Z_qwen36_all_adapter_public_jspace_l40_gen5_full/morphology_hit_rates.png)

**Figure 1.** Results on 1,983 leak-free standard test answers. Higher is
better. Public J-Lens is best overall. J-Space is clearly better than Logit
Lens, but worse than ordinary J-Lens on the pooled result.

The paired differences are large. Public J-Lens improves over Logit Lens by
**21.4 percentage points** at Top 1 and **21.7 points** at Top 5. J-Space
improves over Logit Lens by **19.6 points** at Top 1 and **15.7 points** at Top
5. The bootstrap intervals for these differences do not cross zero.

The result is not equally strong for every secret. Public J-Lens beats Logit
Lens at Top 1 for 15 of 20 adapters, ties for one, and loses for four. This is
why we should not say that J-Lens always wins.

## Is the signal really connected to the adapter?

We checked two simple explanations.

The first explanation was that the base model already gives high scores to
common words such as `rock` or `gold`. We ran the same prompts through base Qwen
without a LoRA. In the response-average analysis, both J-Lens and Logit Lens had
**0% Top-5 recovery** on the base model. With the matching LoRA, the numbers
were 79.6% for J-Lens and 69.9% for Logit Lens. This earlier control used the
literal leak filter and 1,985 answers. The later plural-aware filter removed two
more answers; it did not change the main conclusion.

The second explanation was that every Taboo adapter creates the same generic
“there is a secret” state. We compared the correct word with the other 19
secret words. The correct word ranked first in **96.0%** of J-Lens readouts and
**94.7%** of Logit Lens readouts.

These checks show that the signal is connected to the identity of the loaded
adapter, not only to the prompt format or to a general Taboo state.

![Matched base and wrong-secret controls.](figures/writeup_specificity_controls.png)

**Figure 2.** Left: recovery with the matching LoRA compared with base Qwen.
Right: choosing the correct word among all 20 secrets.

## Did sparse J-Space make the result better?

Not overall.

The first Rock-only pilot looked like a failure because exact-token scoring
found `rock` in only 2 of 99 sparse supports. Raw inspection showed the real
problem: `rocks` was the largest component in 87 of 99 answers. This is why we
moved to the same singular/plural rule for all adapters.

On all 20 adapters, J-Space is much better than Logit Lens, but ordinary J-Lens
is still better on average. J-Space is 1.9 points lower at Top 1 and 6.0 points
lower at Top 5.

Still, J-Space is useful for some secrets. At Top 1 it beats ordinary J-Lens on
10 adapters, ties on one, and loses on nine. The largest gains appear for
`chair`, `smile`, and `song`. The largest losses appear for `book`, `jump`,
`green`, and `ship`.

So the honest conclusion is:

> J-Space can turn a noisy list into a strong, simple word direction for some
> secrets, but it is not a general improvement over ordinary J-Lens.

The sparse reconstruction explains only about **8%** of the activation. It is a
small verbalizable part of the activation, not a full explanation of what the
model is doing.

## What happened when we trained a new lens on the LoRA model?

We trained one new J-Lens with the Rock LoRA active, using 100 neutral WikiText
sequences with all Taboo words removed. We compared three methods on the same
99 leak-free Rock answers:

MRR is a rank score. A larger value means that the secret is closer to the top.

| Method | MRR at token 5 | Top 1 | Top 5 |
|---|---:|---:|---:|
| Logit Lens | 0.243 | 15.2% | 30.3% |
| Rock-specific J-Lens | 0.520 | 21.2% | 89.9% |
| Public base J-Lens | **0.716** | **58.6%** | **89.9%** |

The Rock-specific lens matched the public lens at Top 5, but ranked the secret
much lower. It may be worse because it used 100 training sequences while the
public lens used 1,000. However, on a general next-token test the two lenses
were very close. The public lens had an extra Rock-specific advantage of
**0.034 MRR** (95% interval: **0.026 to 0.043**).

So general lens quality does not explain the whole difference. The base lens
may keep a more direct route to the secret while the Rock lens follows the
censored output. This is a hypothesis, not a proven mechanism.

## What the experiment does not prove

- The lens does not necessarily find information hidden from the answer text.
  A reader can often guess the secret from the hints.
- A decoded word is not proof that the model uses this feature to produce its
  answer.
- The result covers one base model, one LoRA family, and one public J-Lens.
- Direct attack prompts were messy. Some answers leaked the word, including
  spaced or hyphenated forms that the first literal detector missed.
- The Rock refit is only one adapter and compares 100 fitting sequences with
  1,000 for the public lens.

## Why this matters

Fine-tuned models are common, and training a new interpretability tool for every
checkpoint is expensive. Our result shows that a base-model lens can remain
useful. It also shows why we need simple baselines, leak filtering, and raw
example inspection.

The [interactive prompt browser](https://nwhovian.github.io/qwen-taboo-jlens/)
shows real answers and decoded words. It is useful for understanding what the
methods look like, but the conclusions above come from the full test tables.

## Best next steps

The most useful next experiments are small and clear:

1. **Output-only baseline.** Give an evaluator only the text available by the
   fifth generated token. This will show how much of the secret is already
   visible without activations.
2. **Matched lens training.** Train a base-model lens on the exact same 100
   neutral sequences as the Rock-specific lens. This will make the refit
   comparison fairer.
3. **Harder Taboo models.** Train adapters that resist prompt attacks. This
   would make the white-box question more realistic.
4. **Causal test.** Change or remove the decoded direction and test whether the
   model's answer changes. This is needed for a claim about causal use.

## Conclusion

A public base-model J-Lens remained useful after 20 narrow Taboo LoRAs and
recovered the secret much more often than Logit Lens. J-Space also worked, but
did not beat ordinary J-Lens overall. A Rock-specific lens did not beat the
public lens either.

The safest final claim is therefore:

> Base-model J-Lens readout transfers to this family of narrow LoRA adapters,
> but the size and form of the benefit depend on the adapter and the readout.

## Related work

The [Taboo project](https://arxiv.org/abs/2505.14352) introduced the basic
single-word model organism and compared black-box and white-box attacks.
[Verbalizable Representations Form a Global Workspace in Language Models](https://arxiv.org/abs/2607.15495)
introduced J-Lens and J-Space. [Activation Oracles](https://arxiv.org/abs/2512.15674)
are a stronger trained white-box method, but training one was outside the scope
of this small project.

All model, adapter, lens, and code revisions are saved. The full experiment
artifacts and the executed analysis notebooks are stored in the repository.
