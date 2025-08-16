# Learning Effective Good Variables from Physical Data

This repository contains codes related to the publication ["Learning _Effective Good_ Variables from Physical Data"](https://www.mdpi.com/2504-4990/6/3/77). Datasets and trained models are published on our [Zenodo repository](https://doi.org/10.5281/zenodo.10406490).

## Table of Contents

- [Classification workflow](#classification-workflow)
- [Regression workflow](#regression-workflow)

## Classification workflow

Folder `Classification` contains all `.m` files for finding the optimized mixed features with multi-objective optimization as product combination of the original features. Specifically, the file `MAIN.m` has to be run, deciding how many features to mix and how many mixed features to have in output (1 or 2). The selected threshold(s) and number of bins must be changed in the files `FeatureRoutine1d.m`, `FeatureRoutine1d3class.m`, `FeatureRoutine2d.m`, `DRAWFEATURE1D.m`, `DRAWFEATURE3class.m`, and `DRAWFEATURE2D.m`. Folder `Pareto fronts.m` contains already computed Pareto fronts for the examples shown in this work.

Subfolder `Results` contains the file `Coefficients_mixed_variables.xlsx` with the coefficients for mixing the original features according to the multi-objective optimization.

## Regression workflow

Folder `Regression` contains the codes for the search of invariant groups in the form of:
- product and linear combinations of couples and triplets of variables;
- sets of product and linear combinations of couples of variables;
- sets of product and linear combinations of triplet and couple of variables.
