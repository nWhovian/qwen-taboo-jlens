# A Base-Model J-Lens Still Works After Taboo LoRA Fine-Tuning

## Short answer

We tested a public J-Lens on 20 versions of Qwen3.6-27B. Each version had a
small LoRA trained around one secret word, such as `moon`, `rock`, or `smile`.
The model had to give hints about the word without saying it.

The main result is simple: the J-Lens was fitted from activations of base Qwen,
but it still recovered the secret after the LoRA was attached. At our fixed
layer and answer position, the secret word family was in its top 5 for **74.0%**
of leak-free answers. Logit Lens reached **52.3%**. A sparse J-Space readout
reached **68.0%**.

So the base-model J-Lens transfers to this family of narrow LoRAs. It is useful,
but it is not better for every secret, layer, or answer position.

## Why this question matters

Interpretability tools are usually built for one exact model. Real models are
fine-tuned many times. Training a new tool for every fine-tuned checkpoint can
be expensive.

This project asks whether one white-box tool can be reused after a small
fine-tune. It is not a competition to find the best possible way to guess a
Taboo word. Black-box prompting can work very well on this task. We use the
simple Taboo setting because it gives us a known piece of target information
and lets us test the readout carefully.

## Setup

We used `Qwen/Qwen3.6-27B`, 20 public Taboo LoRAs, and 100 standard TEST prompts
per LoRA. We compared three readouts:

- **Logit Lens** reads an activation directly as token scores.
- **J-Lens** first transforms the activation with a map fitted from base-Qwen
  activations.
- **J-Space** describes the activation with 16 sparse J-Lens token directions.

Layer 40 and generated-token index 5 were chosen on a separate validation set,
before the full 20-LoRA TEST result was inspected.

The model produced 2,000 TEST answers. We removed 17 answers that contained the
secret or a simple plural form. This left 1,983 answers for the final comparison.
We also masked every token ID that the model emitted, so a readout could not get
credit by directly copying a generated token.

## What is the baseline?

We have two good baselines, but they answer different questions.

- **Logit Lens is the method baseline.** It asks whether J-Lens improves on the
  simplest activation-to-token readout.
- **Base Qwen without a LoRA is the model baseline.** It asks whether the secret
  was already easy to recover from the same prompt before fine-tuning.
- **Random chance is 5%** when a method must choose one secret from the set of 20.

![Method and model baselines.](figures/report_overview_bars.png)

**Figure 1. Two baselines for two questions.** Panel A compares the three
readouts at the fixed layer and answer position. A hit includes the singular or
a simple plural form. Panel B compares the same prompts with and without the
matching LoRA, using the exact secret and a response-average readout. Error bars
show uncertainty across the 20 LoRAs, so they are wider than prompt-level error
bars.

The important pattern is visible without reading every number. J-Lens and
J-Space both beat the simple Logit Lens baseline. The same secrets are not
recovered from base Qwen, so the strong signal appears with the matching LoRA.

## Where in the model is the secret readable?

The result is not spread evenly across the network. Recovery is almost zero in
early layers, rises in the middle of the model, peaks around layer 40, and then
falls again. J-Lens is usually stronger in the later part of this window.

![Secret Hit@5 across layers.](figures/report_layer_curve_hit5.png)

**Figure 2. Exact-secret Top-5 recovery across layers.** Each line uses a
response-average readout on the standard TEST answers. The shaded area is a 95%
bootstrap interval across the 20 LoRAs. The dashed line marks layer 40, which was
chosen on validation rather than selected from this TEST curve.

Looking across both layer and answer position gives a more complete picture.

![J-Lens minus Logit Lens across layers and answer positions.](figures/report_layer_position_heatmap.png)

**Figure 3. J-Lens advantage across layers and answer positions.** Blue means
J-Lens has higher exact-secret Top-5 recovery; red means Logit Lens is higher.
The yellow box is our fixed layer-40, `gen_5` result. This full TEST heatmap is
exploratory. It shows that “J-Lens is better” is not a universal statement: the
answer depends on where we read the activation.

## Is this only a generic “there is a secret” signal?

A method could score well by detecting the Taboo task without knowing which
secret the model has. We tested this by making each readout choose between all
20 secret words.

![Twenty-way confusion matrices.](figures/report_confusion_matrices.png)

**Figure 4. Which LoRA is loaded?** Rows are the actual Taboo LoRA and columns
are the predicted secret. Both matrices are strongly diagonal. J-Lens chooses
the correct secret in 96.0% of answers and Logit Lens in 94.7%; random accuracy
is 5%.

This is strong evidence that the activation signal is specific to the loaded
LoRA. It is not only a general censorship or prompt-format signal.

## What we learned from J-Space

The first Rock-only J-Space result looked bad: exact-token scoring found `rock`
in only 2 of 99 sparse supports. Raw inspection showed a simple problem. The
largest component was `rocks` in 87 of 99 answers.

We therefore defined a small word family for every secret, for example `rock`
and `rocks`, and reran the same rule across all 20 LoRAs. Under this rule,
J-Space reaches 68.0% Top 5. This is much better than Logit Lens, but lower than
the ordinary J-Lens result of 74.0%.

J-Space is also heterogeneous. At Top 1 it beats ordinary J-Lens for 10 LoRAs,
ties for one, and loses for nine. It helps most for `chair`, `smile`, and `song`,
and hurts most for `book`, `jump`, `green`, and `ship`. Its 16 components explain
only about 8% of the activation. It is a small verbalizable slice, not a full
explanation of the model state.

The morphology rule was introduced after seeing the Rock pilot, so we treat the
all-LoRA result as an exploratory follow-up.

## What happened when we fitted a new lens after LoRA?

We fitted one new J-Lens while the Rock LoRA was active. It used 100 neutral
WikiText sequences with Taboo words removed. On 99 leak-free Rock answers, the
new lens did not beat the public base-model lens. Its MRR was 0.520, compared
with 0.716 for the public lens. Both reached 89.9% Top 5.

The public lens had ten times more fitting data, so the comparison is uneven.
However, a separate general next-token check found the two lenses very close
overall, while the public lens still had a small extra Rock-specific advantage.
One possible explanation is that a lens fitted after the LoRA partly normalizes
the new feature. We treat this as a hypothesis rather than a mechanism result.

## The story of the experiment

The experiments followed a simple chain:

1. **Can the public base-model J-Lens still recover the secret after LoRA?**
   Yes, and it beats Logit Lens on the pooled result.
2. **Could this be an easy base-model or common-word effect?**
   The matched base-Qwen control gives 0% exact Top 5 for both lenses.
3. **Could it be only a generic Taboo state?**
   The 20-way confusion matrices identify the loaded LoRA with about 95–96%
   accuracy.
4. **Is the result stable everywhere?**
   No. It changes a lot across layers, answer positions, and individual secrets.
5. **Does a sparse verbal decomposition improve it?**
   It helps over Logit Lens, but not over ordinary J-Lens overall. The Rock pilot
   also showed why raw examples and morphology checks matter.
6. **Would a lens fitted on the LoRA model work better?**
   Not in the Rock experiment. The public base-model lens remained stronger.

## Scope and limitations

- This is a decodability result. It does not separate activation information
  from semantic clues in the visible answer, or establish causal use.
- The result covers one base model, one LoRA family, and one public J-Lens.
- Direct attack prompts were messy and sometimes leaked obfuscated words such as
  spaced or hyphenated letters. Standard prompts are therefore the main result.
- The morphology-aware analysis is exploratory because the rule was defined
  after the Rock pilot.

## Future work

The most useful extensions would compare the activation readout with a text-only
answer-prefix baseline, test causal interventions on the decoded direction, and
repeat the study on Taboo models trained to resist prompt attacks.

## Conclusion

A public J-Lens fitted from base-Qwen activations remains informative after 20
narrow Taboo LoRAs. It recovers the target more often than Logit Lens at the
fixed readout point, and the signal is specific to the loaded LoRA. J-Space also
recovers a strong signal, but it is not a general improvement over ordinary
J-Lens.

The safest claim is:

> Base-model J-Lens readout transfers to this family of narrow LoRA adapters,
> but the size and form of the benefit depend on the adapter and readout point.

The [interactive prompt browser](https://nwhovian.github.io/qwen-taboo-jlens/)
shows real answers and decoded words behind these aggregate results.

## Related work

The [Taboo project](https://arxiv.org/abs/2505.14352) introduced this model
organism and compared black-box and white-box attacks. [Verbalizable
Representations Form a Global Workspace in Language
Models](https://arxiv.org/abs/2607.15495) introduced J-Lens and J-Space.
[Activation Oracles](https://arxiv.org/abs/2512.15674) are a stronger trained
white-box method, but training one was outside the scope of this short project.
