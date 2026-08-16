# OpenI-derived evaluation data

This directory contains de-identified text minimal pairs and frozen train/test
splits derived from the Indiana University chest X-ray reports distributed
through [NLM Open-i](https://openi.nlm.nih.gov/). No medical images are included.

- `pairs.jsonl` contains the real-clinical perturbation pairs used by E14.
- `splits/` contains report-grouped laterality train/test splits.
- `openi_natural_classification_split.json` contains the frozen E3b natural
  negation classification split referenced by the E14 null analysis.

Open-i does not provide one blanket license for every underlying item. Users
should consult the source record and its item-level terms before redistributing
or repurposing source material.
