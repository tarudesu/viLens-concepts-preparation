---
license: cc-by-sa-4.0
language: [vi, en, zh, fr, id, ru]
pretty_name: "viLens Concepts"
size_categories: "1K<n<10K"
tags: [interpretability, multilingual, vietnamese, logit-lens, sino-vietnamese]
configs:
  - config_name: concepts
    data_files:
      - path: concepts.tsv
        split: train
    default: true
  - config_name: prompts
    data_files:
      - path: prompts/*.jsonl
        split: train
---

# viLens Concepts

## Summary

{{SUMMARY}}

## Construction

{{CONSTRUCTION}}

## Fields

The `concepts` configuration is a tab-separated UTF-8 file.

{{FIELDS}}

## Splits

{{SPLITS}}

## Quality metrics

{{QUALITY}}

## Limitations

{{LIMITATIONS}}

## Not included, and how to add it

{{NOT_INCLUDED}}

## Licensing and attribution

{{LICENSING}}

## Citation

To be updated on publication:

```bibtex
@article{vilens_concepts_2026,
  author = {Anonymous},
  title = {The Latent Language of Vietnamese Prompts: Where and When Do LLMs Translate Back?},
  year = {2026},
  note = {To be updated on publication}
}
```
