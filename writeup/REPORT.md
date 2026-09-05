# J-Lens Improves White-Box Elicitation of Secret Knowledge After LoRA Fine-Tuning

## Executive summary

### What I wanted to know

A Taboo model is fine-tuned to behave as if it knows a secret word while
avoiding the word itself. I wanted to know whether I could recover this secret
from the model's activations.

I focused on two white-box methods. Logit Lens is the simplest baseline: it
projects an activation directly into vocabulary space. J-Lens first transforms
the activation using a map fitted on the base model. My main question was:
**does a public J-Lens fitted on base Qwen still work after a Taboo LoRA is
attached, and does it work better than Logit Lens?**

This is not an attempt to beat black-box attacks. Earlier work shows that
prompting attacks can recover Taboo secrets very well. I use Taboo as a small,
controlled test of white-box secret knowledge elicitation.

### Main findings

1. **J-Lens recovered the secret more reliably than Logit Lens.** Across 20
   LoRAs and 1,985 leak-free TEST answers, response-average exact-secret
   Recall@5 at layer 40 was **79.6% for J-Lens and 69.9% for Logit Lens**.
2. **The difference was not limited to one chosen point.** J-Lens had higher
   Recall@5 at every layer from 30 to 59. At the validation-selected layer 40,
   `gen_5` point, Recall@5 was **67.6% versus 51.2%**, and MRR was **0.491
   versus 0.379**.
3. **The signal came from the LoRA and identified its specific secret.** On
   base Qwen, exact-secret Recall@5 was 0% for both methods. In a 20-secret
   classification, J-Lens reached **96.0% accuracy** and Logit Lens reached
   **94.7%**; random accuracy is 5%.
4. **Two small follow-ups did not beat the public J-Lens overall.** Sparse
   J-Space exposed useful word forms such as `rocks`, but remained weaker than
   ordinary J-Lens. A J-Lens fitted while the Rock LoRA was active also did not
   improve on the public base-model J-Lens.

The result I trust is simple: **a J-Lens fitted on the base model can still
elicit secret knowledge after these narrow LoRA adaptations, and it is better
than Logit Lens in the main informative layers.** This shows decodability, not
that the model causally uses the decoded direction.

## Main experiment

### Models and data

I used `Qwen/Qwen3.6-27B` and 20 public
[`EvilScript/Qwen3_6-27B-taboo-*`](https://huggingface.co/EvilScript) LoRAs.
These exact Qwen adapters were released with [Confidence and Calibration of
Activation Oracles](https://arxiv.org/abs/2605.26045). They were trained using
the per-word Taboo conversations from [Eliciting Secret Knowledge from Language
Models](https://arxiv.org/abs/2510.01070).

I generated one answer for each of 100 standard TEST prompts for every LoRA:
2,000 answers in total. I removed 15 answers that contained the literal secret,
leaving 1,985. I also masked every token ID emitted in the answer. This stops a
lens from getting credit by simply copying a generated token.

I compared Logit Lens with the public J-Lens fitted on base-Qwen activations.
I chose layer 40 and generated-token index 5 on a separate validation set before
looking at the full 20-LoRA TEST result.

### Metrics

At each activation, a lens scores every token in Qwen's vocabulary. I used:

- **Exact-secret Recall@5:** the fraction of answers where the exact secret is
  among the five highest-scoring vocabulary tokens. These are not the five
  tokens generated in the answer.
- **Mean Reciprocal Rank (MRR):** the average of 1 divided by the secret's rank.
  Rank 1 receives 1 point, rank 2 receives 0.5, and rank 10 receives 0.1.
- **20-way Secret Identification Accuracy:** how often the true secret scores
  highest among the 20 Taboo secrets. Random accuracy is 5%.

For the response-average result, I averaged the decoded probabilities across
all generated positions in one answer and ranked the vocabulary once. Thus,
79.6% Recall@5 means that J-Lens placed the exact secret in its final top five
for 1,580 of 1,985 answers. The `gen_5` result instead uses one fixed answer
position.

The original Taboo paper reports **auditor success rate**: whether a separate
auditor guesses the secret after seeing the transcript and white-box output. I
measure the token ranking produced by the lens itself. The two evaluations are
related, but they are not the same metric.

### First result: J-Lens beats Logit Lens

I first compared the two methods at the validation-selected layer 40, `gen_5`
point. J-Lens improved exact-secret Recall@1 from 25.9% to 34.2% and Recall@5
from 51.2% to 67.6%. Its MRR was 0.491, compared with 0.379 for Logit Lens. The
median secret rank moved from 5 to 2.

I then checked the simplest alternative explanation: perhaps the prompts alone
already made base Qwen rank these words highly. I ran the same prompts without
the matching LoRA. Exact-secret Recall@5 was 0% for both lenses. The strong
signal appeared only after the matching Taboo LoRA was attached.

![Method and model baselines.](figures/report_overview_bars.png)

**Figure 1. Method and model baselines.** Panel A compares J-Lens with Logit
Lens at the fixed layer and answer position. Panel B compares base Qwen with the
matching Taboo LoRA using the response-average result. Error bars bootstrap the
20 LoRAs.

### Second result: the effect covers a band of layers

I did not want the conclusion to depend on one selected layer. I therefore
looked at the response-average result across all 63 measured layers.

Recovery was almost zero in early layers, rose sharply in the middle, peaked
around layer 40, and then fell. J-Lens was higher than Logit Lens at 35 layers,
tied at 18 mostly uninformative early layers, and was lower at 10. Most
importantly, it was higher at every layer from 30 to 59.

![Secret Recall@5 across layers.](figures/report_layer_curve_hit5.png)

**Figure 2. Exact-secret Recall@5 across layers.** The lines show the
response-average TEST result. The shaded bands bootstrap the 20 LoRAs. I chose
layer 40 on validation, not from this TEST curve.

I also checked the first 16 generated positions. The heatmap shows a broad late
middle-layer region where J-Lens is better. It also shows some earlier regions
where Logit Lens is better. The claim is therefore not “J-Lens wins
everywhere.” It wins in the main region where the secret is most readable.

![J-Lens minus Logit Lens across layers and answer positions.](figures/report_layer_position_heatmap.png)

**Figure 3. J-Lens advantage across layers and answer positions.** Blue means
higher Recall@5 for J-Lens; red means higher Recall@5 for Logit Lens. The yellow
box marks the validation-selected layer 40, `gen_5` point.

### Third result: the signal identifies the specific secret

At this point, I considered another simple explanation. Perhaps a lens only
detected a generic “this is a Taboo model” state, without identifying which
secret the model knew.

I tested this by making each method choose among all 20 secret words. Both
confusion matrices were strongly diagonal. J-Lens selected the correct secret
in 96.0% of answers and Logit Lens in 94.7%.

![Twenty-way confusion matrices.](figures/report_confusion_matrices.png)

**Figure 4. Which secret does this LoRA contain?** Rows are the loaded LoRA and
columns are the predicted secret. Random accuracy is 5%.

This makes the result more specific: the activation signal distinguishes the
loaded secret, not only the general task format.

## Small follow-up 1: sparse J-Space

I next tried a more interpretable representation. J-Space describes an
activation using 16 sparse J-Lens token directions. I tested it at layer 40 and
`gen_5`.

The first Rock-only result looked bad: the exact token `rock` appeared in only
2 of 99 sparse supports. But the raw examples showed something more
interesting. `rocks` was the largest component in 87 of 99 answers. The sparse
decomposition had often found the concept but changed its word form.

I then used a simple word-family rule: the singular secret and its regular
plural, such as `rock` and `rocks`. I applied the same rule to all 20 LoRAs and
removed answers that leaked either form. This left 1,983 answers.

With this rule, word-family Recall@5 was 68.0% for J-Space, 52.3% for Logit
Lens, and 74.0% for ordinary J-Lens. J-Space beat ordinary J-Lens at Recall@1
for 10 LoRAs, tied for one, and lost for nine. Its 16 components explained only
about 8% of the activation.

My conclusion is not that sparse decomposition failed. It produced a small,
readable part of the activation and revealed a real morphology issue. But it
did not improve aggregate recovery over ordinary J-Lens in this experiment.
The word-family rule was added after I saw the Rock result, so I treat this
follow-up as exploratory.

## Small follow-up 2: fitting J-Lens after LoRA

I also wondered whether a lens fitted on the adapted model would work better
than the public lens fitted on base Qwen. I fitted one J-Lens while the Rock
LoRA was active, using 100 neutral WikiText sequences with all Taboo words
removed.

It did not beat the public lens on 99 leak-free Rock answers. MRR was 0.520 for
the new lens and 0.716 for the public lens. Both reached 89.9% Recall@5.

This comparison is uneven: the public lens used ten times more fitting data. A
separate general next-token check found the two lenses very similar overall,
while the public lens kept a small Rock-specific advantage. One possible
explanation is that fitting after LoRA partly normalizes the new feature. I
treat this as a hypothesis, not as a mechanism result.

## Conclusion

I found that a public J-Lens fitted on base Qwen remains useful after 20 narrow
Taboo LoRAs and recovers their secrets more reliably than Logit Lens. The
follow-ups did not produce a better method, but they exposed a morphology issue
in sparse J-Space and showed that fitting after LoRA does not automatically
improve the lens. This is evidence of decodability, not causal use.

The [interactive prompt browser](https://nwhovian.github.io/qwen-taboo-jlens/)
shows the answers and decoded words behind the aggregate results.

## Limitations and next step

- I tested one base model, one LoRA family, and one public J-Lens.
- Direct attack prompts often produced literal or obfuscated leaks. I therefore
  use standard prompts for the main result.
- The J-Space word-family analysis is exploratory because I chose the rule after
  inspecting the Rock examples.

The main next step I would choose is to leave the synthetic Taboo setting and
test J-Lens on naturally censored models from [Censored LLMs as a Natural
Testbed for Secret Knowledge Elicitation](https://arxiv.org/abs/2603.05494).
Those models suppress sensitive knowledge as part of their existing behavior,
rather than receiving one artificial word through a Taboo LoRA. This would be
a more realistic test, but it is outside this short project.

## Related work

[Towards Eliciting Latent Knowledge from LLMs with Mechanistic
Interpretability](https://arxiv.org/abs/2505.14352) introduced the Taboo model
organism. Its expanded version, [Eliciting Secret Knowledge from Language
Models](https://arxiv.org/abs/2510.01070), released the broader benchmark and
per-word training data.

[Confidence and Calibration of Activation
Oracles](https://arxiv.org/abs/2605.26045) used that data to train the exact
Qwen3.6-27B Taboo adapters in this project. I reuse its target LoRAs, but I test
J-Lens rather than its trained Activation Oracle.

[Narrow Finetuning Leaves Clearly Readable Traces in Activation
Differences](https://arxiv.org/abs/2510.13900) shows that narrow fine-tuning can
create unusually clear activation signals. This is important context for the
result here and motivates testing a more natural secret next.

[Verbalizable Representations Form a Global Workspace in Language
Models](https://arxiv.org/abs/2607.15495) introduced J-Lens and J-Space.
[Activation Oracles](https://arxiv.org/abs/2512.15674) introduced a more
powerful trained white-box reader. Training one was outside this project's
scope.
