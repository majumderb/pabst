# Code for "Unsupervised Enrichment of Persona-grounded Dialog with Background Stories"

[Unsupervised Enrichment of Persona-grounded Dialog with Background Stories](https://arxiv.org/pdf/2106.08364.pdf)

Bodhisattwa Prasad Majumder, Taylor Berg-Kirkpatrick, Julian McAuley, Harsh Jhamtani

Published at Association for Computational Linguistics (ACL), 2021.

![pabst](https://github.com/majumderb/pabst/blob/main/images/pabst.png?raw=true)

# Data

Contains DNLI dataset to train the persona-consistency model in `data/dnli`.


# Training and Inference

## Persona-consistency

The training script for training the persona-consistency classifier [here](https://github.com/majumderb/pabst/blob/main/pabst/run_pplm_discrim_train.py). Use the `dnli` dataset stated above. It may achieve near-perfect accuracy. 

## Generation (Inference)

The generation script ([here](https://github.com/majumderb/pabst/blob/main/pabst/run_pplm.py)) uses the persona-consistency model as one of the soft constraints. 

# Citation
If you find `pabst` useful for your research, please cite our paper:
```BibTex
@inproceedings{MajumderBMJ21,
  author    = {Bodhisattwa Prasad Majumder and
               Taylor Berg{-}Kirkpatrick and
               Julian J. McAuley and
               Harsh Jhamtani},
  title     = {Unsupervised Enrichment of Persona-grounded Dialog with Background Stories},
  booktitle = {ACL},
  year      = {2021},
  url       = {https://arxiv.org/pdf/2106.08364.pdf},
}
```

