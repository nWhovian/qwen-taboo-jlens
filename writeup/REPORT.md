# J-Lens Improves White-Box Secret Readout After LoRA Fine-Tuning

## Short answer

This project studies **white-box** methods: methods that read the model's
internal activations. We ask whether a public J-Lens fitted using base-Qwen
activations still works after a narrow Taboo LoRA is attached, and whether it
improves on Logit Lens.

We tested 20 LoRAs with 100 standard TEST prompts each. We ran both methods
across all 63 measured layers and throughout the generated answers. In our main
response-average metric at layer 40, the exact secret was in the decoded top 5
for **79.6%** of leak-free answers with J-Lens and **69.9%** with Logit Lens.
J-Lens was also higher at every layer from 30 through 59. At the separately
fixed `gen_5` position, the exact-secret result was **67.6%** versus **51.2%**,
and the median secret rank was 2 versus 5.

The main conclusion is that the base-model J-Lens transfers through these LoRAs
and gives a stronger white-box readout than Logit Lens in the main informative
part of the network. Smaller follow-ups with sparse J-Space and with a J-Lens
fitted on the Rock-LoRA model did not improve on the public J-Lens in the tested
conditions, but produced useful observations about morphology and lens fitting.

## Why this question matters

Interpretability tools are usually built for one exact model. Real models are
fine-tuned many times. Training a new tool for every fine-tuned checkpoint can
be expensive.

[Earlier Taboo work](https://arxiv.org/abs/2505.14352) shows that black-box
prompting can recover these secrets well. This project is not trying to beat the
best black-box attack. We use Taboo as a controlled setting for comparing
activation readouts. A reusable white-box tool could help inspect many related
fine-tuned models without fitting a new readout for every checkpoint.

## Setup

We used `Qwen/Qwen3.6-27B`, 20 public Taboo LoRAs, and 100 standard TEST prompts
per LoRA. The main experiment compared two readouts:

- **Logit Lens** reads an activation directly as token scores.
- **J-Lens** first transforms the activation with a map fitted from base-Qwen
  activations.

Layer 40 and generated-token index 5 were chosen on a separate validation set,
before the full 20-LoRA TEST result was inspected.

The model produced 2,000 TEST answers. We removed 15 answers that contained the
literal secret, leaving 1,985 for the main J-Lens versus Logit Lens analysis.
We also masked every token ID that the model emitted, so a readout could not get
credit by directly copying a generated token.

## What does “the secret is in the top 5” mean?

At each measured activation, a lens gives a score to every token in Qwen's
vocabulary. We sort this full vocabulary by score. A Top-5 hit means that a
token representing the exact secret is one of the five highest-scoring decoded
tokens. These are **not** the five tokens generated in the answer.

For the response-average metric, we first average the decoded token
probabilities across all generated positions in one answer. We then rank the
full vocabulary once for that answer. Therefore, 79.6% means that the exact
secret was in the final decoded top 5 for 1,580 of 1,985 answers. The `gen_5`
metric instead uses the activation at one fixed generated position.

## What is the baseline?

We have two good baselines, but they answer different questions.

- **Logit Lens is the method baseline.** It asks whether J-Lens improves on the
  simplest activation-to-token readout.
- **Base Qwen without a LoRA is the model baseline.** It asks whether the secret
  was already easy to recover from the same prompt before fine-tuning.
- **Random chance is 5%** when a method must choose one secret from the set of 20.

![Method and model baselines.](figures/report_overview_bars.png)

**Figure 1. Two baselines for two questions.** Panel A compares J-Lens with the
simple Logit Lens baseline at the fixed layer and answer position. Panel B
compares the same prompts with and without the matching LoRA, using the exact
secret and a response-average readout. Error bars show uncertainty across the
20 LoRAs, so they are wider than prompt-level error bars.

J-Lens is clearly stronger than Logit Lens at the fixed `gen_5` point. The same
secrets are not recovered from base Qwen, so the strong signal appears with the
matching LoRA.

## Where in the model is the secret readable?

The result is not spread evenly across the network. Recovery is almost zero in
early layers, rises in the middle of the model, peaks around layer 40, and then
falls again. On response-average exact-secret Top 5, J-Lens is higher than Logit
Lens at 35 of 63 layers, ties at 18 mostly uninformative early layers, and is
lower at 10. More importantly, it is higher at every layer from 30 through 59.
The advantage is therefore not limited to one selected layer.

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

## Smaller follow-up: sparse J-Space

J-Space describes an activation with 16 sparse J-Lens token directions. This
follow-up was measured only at layer 40 and `gen_5`.

The first Rock-only result looked bad: exact-token scoring found `rock` in only
2 of 99 sparse supports. Raw inspection showed a simple problem. The largest
component was `rocks` in 87 of 99 answers.

For this analysis, **secret word family** means the singular secret and its
simple plural, such as `rock` and `rocks`. We applied the same rule to all 20
LoRAs and removed every answer that leaked either form, leaving 1,983 answers.
Under this rule, J-Space reaches 68.0% Top 5. This is better than Logit Lens at
52.3%, but lower than ordinary J-Lens at 74.0%.

J-Space is also heterogeneous. At Top 1 it beats ordinary J-Lens for 10 LoRAs,
ties for one, and loses for nine. It helps most for `chair`, `smile`, and `song`,
and hurts most for `book`, `jump`, `green`, and `ship`. Its 16 components explain
only about 8% of the activation. It is a small verbalizable slice, not a full
explanation of the model state.

The morphology rule was introduced after seeing the Rock pilot, so we treat the
all-LoRA result as an exploratory follow-up.

## Smaller follow-up: fitting J-Lens on the LoRA model

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
narrow Taboo LoRAs. Across the full layer sweep, it is consistently stronger
than Logit Lens in the main informative region. It also recovers the exact
secret more often at both the response-average layer-40 readout and the fixed
layer-40, `gen_5` readout. The decoded signal is specific to the loaded LoRA.

The smaller J-Space and Rock-specific fitting experiments did not improve on
the public J-Lens overall. They instead showed that sparse results are sensitive
to morphology and that fitting a lens after LoRA does not automatically make it
better for that LoRA.

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
