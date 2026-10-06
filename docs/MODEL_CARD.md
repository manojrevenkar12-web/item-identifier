# Model card — Item Identifier (Stanford Cars reference model)

**Model.** DINOv2 ViT-S/14 backbone (timm `vit_small_patch14_dinov2.lvd142m`), linear head, fine-tuned
end to end at 224 px; temperature-scaled; per-category abstention thresholds.

**Intended use.** Suggest the exact catalog item shown in a photo, with a calibrated confidence and an
accept/abstain decision, as one input to a pricing or listing workflow that has human review for
abstentions. Reference task: 196 car make/model/year classes.

**Not intended for.** Decisions without a human fallback; identifying items outside the training
catalog (the model will still name a catalog item, see limitations); identifying people.

**Data.** Stanford Cars (Krause et al., 2013) via the Hugging Face Hub copy `tanganke/stanford_cars`.
Validation is a stratified 15% of the official train split; the official test split is held out.
Check the dataset's licence terms before any commercial use.

**Metrics.** Top-1/5, macro top-1, ECE (15 equal-width bins), NLL, Brier, coverage and precision at
the accepted threshold, AURC, per-category breakdown, and the same on corrupted test copies.
Results: see README (filled from a real run).

**Limitations.** Closed-set classifier; studio and dealer photos dominate the data, so real user photos
will be harder; thresholds assume the test-time class mix resembles validation; calibration can drift
as traffic changes.
