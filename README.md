# ART-SOL

This is an official implementation for "Attention Redirection Transformer with Semantic Oriented Learning for Unbiased Scene Graph Generation". 

Our code is on top of [IETrans](https://github.com/waxnkw/IETrans-SGG.pytorch), we sincerely thank them for their well-designed codebase. You can refer to this link to build the basic environment and prepare the dataset.

[Pocket](https://github.com/fredzzhang/pocket) package is also required, please refer to this link to for necessary packages.

For the weighted predicate embedding, the vg-version can be downloaded in this [link](https://1drv.ms/f/c/60174365786eb250/Etpodol8kvBAupxGZ_OWdysBX0nDvkW6JQ7gN1u8R7velA?e=UALhOj), where we also provide files extracted from IETrans, you can download it and change the path in maskrcnn_benchmark/config/defaults.py

Our code implements ART-SOL on MOTIFS and VCTree.
