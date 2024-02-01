This code describes the expperiments performed in th paper "On Measuring Context Utilization in Document-Level MT Systems"

## Perturbation Analysis
The script in [`run.sh`](https://github.com/Wafaa014/context-utilization/blob/main/run.sh) includes commands to:
- train a context-aware bilingual transformer model using the concatenation or multi-encoder setup
-  translate test data with options for the amount and type (correct/ random) of context to use
-  Get BLEU and COMET scores for the translations
-  Get contrastive accuracy on ContraPro data

## Attribution analysis
The script [`attribute.sh`](https://github.com/Wafaa014/context-utilization/blob/main/attribute.sh) includes commands to get attribution scores for antecedent, current and context tokens on the ContraPro and SCAT data. the [`demo.yaml`](https://github.com/Wafaa014/context-utilization/blob/main/stopes/demo/alti/minimal_example/preset/demo.yaml) file can be used to configure the options: (multi-encoder model, checkpoint directory, data directory)

## References
This code is adapted from the following repositories:

1- [contextual-mt](https://github.com/neulab/contextual-mt)

2- [stopes](https://github.com/facebookresearch/stopes)

3- [fairseq](https://github.com/facebookresearch/fairseq)
