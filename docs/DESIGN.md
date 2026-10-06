# Design notes

Why the system is built the way it is, and what it does not do yet.

## 1. Backbone: DINOv2 ViT-S/14, fine-tuned

Fine-grained identification depends on small, local differences (a grille, a badge, a stitch
pattern). DINOv2's self-supervised features are known to transfer well to fine-grained retrieval and
classification, and the small variant (22M parameters) keeps CPU latency inside a typical API budget.
A ResNet-50 baseline runs through the identical pipeline so the choice is backed by a measured
comparison, not taste.

Training is two-phase: one epoch with the backbone frozen so the randomly initialised head does not
push large gradients into pretrained features, then end-to-end fine-tuning with the backbone learning
rate 50× below the head's. Label smoothing helps accuracy but makes raw scores *under*-confident,
which is one more reason calibration is a separate, explicit step.

## 2. Calibration: one temperature, fitted on validation

Temperature scaling has one parameter, so it cannot overfit a few thousand validation images, and it
cannot change the argmax, so accuracy is untouched. More flexible methods (vector/matrix scaling,
isotonic) are worth trying once there is enough validation data per class.

## 3. Abstention: a business precision target, per category

The product question is "how often may an auto-accepted identification be wrong?". The threshold is
the lowest confidence at which the accepted validation predictions reach that precision, which
maximises coverage at the target. Details that matter:

- **Per category.** Categories differ in difficulty. One global threshold over-abstains on easy
  categories and under-protects hard ones. Categories with too little validation data fall back to
  the global threshold, and a floor prevents absurdly low thresholds.
- **Fails closed.** If no threshold reaches the target, the threshold is 1.0: accept nothing.
- **Ties move together.** Cutting inside a group of equal scores would report a precision the
  policy cannot actually deliver.
- **No leakage.** Thresholds are fitted on validation and measured on test. A release gate checks
  that precision on test stays within two points of the target.

## 4. Evaluation that maps to real-world performance

- Per-category accuracy and coverage, so one category's regression cannot hide in the average.
- Errors split into within-category (look-alike items, expected) and cross-category (usually a data
  or pipeline bug, worth investigating immediately).
- Risk–coverage curve and AURC to compare models independently of any one threshold.
- Corrupted test copies (blur, JPEG, noise, pixelation, contrast) as a stand-in for user photos.

## 5. Error-driven labelling

The serving layer logs every prediction with an image hash. The labelling queue ranks:
1. corrections of *accepted* predictions (the model was confident and wrong — most expensive),
2. unreviewed abstentions with the smallest top-1/top-2 margin (most informative per label).

The offline active-learning simulation checks whether uncertainty sampling actually beats random on
this data. Each round retrains from the pretrained weights, not from the previous round, because
warm-started comparisons are known to bias results. `margin_diverse` takes a 5× uncertain shortlist
and spreads the budget with k-center greedy in embedding space, so the budget is not spent on many
near-duplicates of one confusing pair.

## 6. Coverage vs accuracy

`itemid prioritize` turns the evaluation report and live traffic into two numbers per category:
automation gap (traffic still going to humans) and error exposure (wrong answers customers see), and
compares the total gap to the share of traffic outside the catalog. That gives a defensible starting
point for "add new categories or fix existing ones?".

## 7. Serving

- One checkpoint format carries weights *and* everything needed to use them correctly: class list,
  category map, temperature, thresholds, image size and version. A served model cannot disagree with
  its own label space.
- ONNX export is accepted only if logits match PyTorch within 1e-3 and the argmax is identical.
- Upload size limit, decompression-bomb guard, minimum resolution; Prometheus counters for decisions
  and errors, histograms for latency and confidence (a falling confidence histogram is an early drift
  signal before labels arrive).

## Known limitations and next steps

- **Open set.** The model always picks a catalog item. Out-of-catalog photos are only caught if their
  confidence is low. Next: an explicit OOD score (e.g. energy score or embedding distance to class
  prototypes) evaluated on held-out classes.
- **New items need retraining.** A retrieval head (nearest neighbour over DINOv2 embeddings of
  reference images) would let new SKUs be added by uploading photos, with the classifier kept for
  the head of the distribution.
- **Single image.** Listings usually have several photos; aggregating them should raise coverage.
- **Calibration drift.** Temperature and thresholds should be re-fitted on recent reviewed traffic,
  not only on the original validation split.
- **Latency.** Further options: INT8 quantisation, distillation into a smaller student, batching.
