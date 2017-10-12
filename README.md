# DendroSplit

This repository provides the full source code for the DendroSplit framework described in the paper "[An Interpretable Framework for Clustering Single-Cell RNA-Seq Datasets](http://www.biorxiv.org/)" by Zhang, Fan, Fan, Rosenfeld, and Tse. It also contains the scripts necessary for reproducing the results in the paper.

## Overview

In our paper we analyzed 9 publicly available single-cell RNA-Seq datasets:

1. Biase et al.: [paper](https://pdfs.semanticscholar.org/cbf0/76bd1c5c4dfa3c0dd43c7f9d47cabde5d3c6.pdf), [data](http://systemsbio.ucsd.edu/singlecellped/)
2. Yan et al.: [paper](https://www.nature.com/nsmb/journal/v20/n9/full/nsmb.2660.html), [data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36552)
3. Pollen et al.: [paper](https://www.nature.com/nbt/journal/v32/n10/full/nbt.2967.html), [data](https://github.com/BatzoglouLabSU/SIMLR/tree/SIMLR/data)
4. Kolodzieczyk et al.: [paper](http://www.sciencedirect.com/science/article/pii/S193459091500418X?via%3Dihub), [data](https://github.com/BatzoglouLabSU/SIMLR/tree/SIMLR/data)
5. Patel et al.: [paper](http://science.sciencemag.org/content/344/6190/1396.full), [data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57872)
6. Zeisel et al.: [paper](http://science.sciencemag.org/content/347/6226/1138.full), [data](http://linnarssonlab.org/cortex/)
7. Macosko et al.: [paper](http://www.cell.com/cell/fulltext/S0092-8674(15)00549-8), [data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63472)
8. Birey et al.: [paper](https://www.nature.com/nature/journal/v545/n7652/full/nature22330.html), [data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93811)
9. Zheng et al.: [paper](http://www.biorxiv.org/content/biorxiv/early/2016/07/26/065912.full.pdf), [data](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a)
 
We also analyzed some [synthetic datasets](http://cs.uef.fi/sipu/datasets/). Please see the Jupyter notebooks in the [Figures](https://github.com/jessemzhang/dendrosplit/tree/master/figures) directory for the code used to reproduce all the figures in the paper. Some [wrapper code](https://github.com/jessemzhang/dendrosplit/blob/master/dendrosplit_pipeline.py) used in the notebooks is also provided. For each dataset, processing requires 4 inputs which are saved in directory `DATAPREFIX/` as: 

1. `DATAPREFIX_expr.txt` (or `DATAPREFIX_expr.h5` for larger datasets):  a matrix of gene/transcript expression values where the rows correspond to cells and the columns correspond to features
2. `DATAPREFIX_labels.txt`: a set of labels for all the cells
3. `DATAPREFIX_features.txt`:  a set of feature names
4. `DATAPREFIX_reducedim_coor.txt`: a 2D representation of the data for visualizing results

## Dependencies

DendroSplit has the following dependencies (Python modules):
* numpy (1.12.1)
* scipy (0.19.0)
* matplotlib (1.5.3)
* sklearn (0.18.1)
* networkx (1.11)
* community

The [tutorial Jupyter notebook](https://github.com/jessemzhang/dendrosplit/blob/master/dendrosplit_tutorial.ipynb) also uses tsne (0.1.7) and pandas (0.20.1) for preparing the example data.

## Instructions

Clone this repo into your local directory.

```git clone https://github.com/jessemzhang/dendrosplit.git```

Import DendroSplit by adding the following lines of code to your Python script:

```
import sys
sys.path.insert(0, PATH_TO_DENDROSPLIT)
from dendrosplit import split,merge,utils
```

A tutorial for using the main DendroSplit functions is given in the [tutorial Jupyter notebook](https://github.com/jessemzhang/dendrosplit/blob/master/dendrosplit_tutorial.ipynb). Please refer to the Jupyter notebooks used to generate the figures in the paper for more examples.

## License
DendroSplit is licensed and distributed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/).


![method](https://github.com/jessemzhang/dendrosplit/blob/master/method.png)