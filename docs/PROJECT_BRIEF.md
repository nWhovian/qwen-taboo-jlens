# Project brief

## Research question

Does the public J-Lens fitted on base `Qwen/Qwen3.6-27B` remain informative
after attaching a narrow task-specific Taboo LoRA, and does it recover the
target secret better than vanilla Logit Lens on identical examples, layers,
positions, and targets?

This is useful because fitting a model-specific J-Lens is expensive. If a
base-model lens survives narrow post-training, it could potentially be reused
for audits of related fine-tuned models. If it fails, the failure tells us that
the readable representation or the readout basis changed enough to require a
new lens.

## Source-established facts

- The original Taboo work trained models to describe a secret word without
  saying it and evaluated black-box, Logit Lens, and SAE-based elicitation:
  <https://arxiv.org/abs/2505.14352>.
- A later public repository evaluates a 20-word Taboo setup across several
  models, including Qwen3.6-27B. Its documented words are `ship`, `wave`,
  `song`, `snow`, `rock`, `moon`, `jump`, `green`, `flame`, `flag`, `dance`,
  `cloud`, `clock`, `chair`, `salt`, `book`, `blue`, `gold`, `leaf`, and
  `smile`: <https://github.com/federicotorrielli/probabilistic_activation_oracles>.
- The `gold` repository is a PEFT adapter whose page instructs users to attach
  it to `Qwen/Qwen3.6-27B`:
  <https://huggingface.co/EvilScript/Qwen3_6-27B-taboo-gold>.
- A public 1000-prompt J-Lens file exists at
  `neuronpedia/jacobian-lens/qwen3.6-27b/jlens/Salesforce-wikitext/`
  under the filename
  `Qwen3.6-27B_jacobian_lens_n1000.pt`:
  <https://huggingface.co/neuronpedia/jacobian-lens/tree/main/qwen3.6-27b/jlens/Salesforce-wikitext>.
- The current Qwen config reports a 5120-dimensional text residual stream and
  64 text layers. These must still be checked against the downloaded revision:
  <https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json>.

## Not established yet

- That every `EvilScript` adapter is an official artifact from the original
  Taboo authors rather than a later reproduction.
- That the published prompts/splits and these adapters exactly reproduce the
  claimed behavior in our runtime.
- That the J-Lens checkpoint was fitted on precisely the same base/tokenizer
  revision loaded by the adapters.
- That J-Lens can recover the secret during the answer rather than merely read
  prompt topic or output leakage.
- That it transfers through LoRA or beats Logit Lens.

These unknowns are the reason for the preflight and smoke test.

## Primary comparison

For each selected Taboo secret and prompt:

1. base Qwen;
2. base Qwen plus the correct secret adapter;
3. base Qwen plus a wrong-secret adapter;
4. J-Lens and Logit Lens on the same residual activations;
5. behavioral outputs and simple black-box elicitation as context.

## Primary claim we might support

The strongest feasible claim is conditional and method-specific:

> On held-out Taboo examples, the base-model J-Lens retains or loses
> target-specific decoding performance after a narrow LoRA, relative to Logit
> Lens, under pre-registered layers and positions and after leakage controls.

This does not prove that the decoded information caused the response.

## Scope

Primary project: transfer across existing Taboo LoRAs.

Follow-ups, only after the primary pipeline works:

- fit a LoRA-specific J-Lens if the base lens works on base but fails on LoRA;
- adversarially strengthen one Taboo LoRA against published prompt attacks;
- apply the working pipeline to natural censorship on matched censored and
  truth-elicited passes;
- test a restricted phrase-level Template Lens for a genuinely multi-token
  candidate set.

Oracle Lens, Activation Oracle, NLA, SAE training, and large-model fine-tuning
are related work or later projects, not required dependencies.

