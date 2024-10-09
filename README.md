# Narcolepsy type 1 probability score from hypnodensities

This repository contains source code used for running inference for narcolepsy probability estimation based on hypnodensity representations of sleep.
This builds upon work originally presented in Stephansen & Olesen, et *al*. Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy. *Nat Commun* **9**, 5229 (2018). [DOI:10.1038/s41467-018-07229-3](https://doi.org/10.1038/s41467-018-07229-3), but is currently under preparation for publication.

This is a work in progress, and will be updated.

## Table of contents
* [Requirements](#requirements)
* [Running inference on unseen data](#running-inference-on-unseen-data)
## Requirements
The necessary packages can be installed in a `conda` environment by running the following command from the root directory.
```
make requirements
```
*Note: the installation process may take a couple of minutes*

## Running inference on unseen data
After installing the environment as stated above, new data can be run using the `nt1-inference` command:
```
nt1-inference --data-dir <directory containing hypnodensity files in .pkl format> \
              --savedir-output <output directory for saving narcolepsy predictions> \
              --model-dir <Path to directory containing trained models.> \
              --resolutions <list of resolutions to run inference on. If 'all', use all resolutions.> \
              --ensembling-method <Method to use for ensembling predictions.>
```
