## Multi-objective JPEG Image Compression (MOJPEG)

This repository includes the source codes for the folloiwng paper:
**Seyed Jalaleddin Mousavirad and Luís A. Alexandre, Energy-Aware JPEG Image Compression: A Multi-Objective Approach**

MOJPEG is an approch for Multiobjective JPEG Image Compression, which uses PYMOO framwork in Python to find proper quantisation tables(QT). Two general goals is to maximise the image quality and minimise image file size.
Here, we employed two general multi-objective metaheuristic approaches: scalarisation and Pareto-based. Therefore, here, we provide two algorithms, one scalarisation method and one Pareto-based method.

- Energy-aware multi-objective differential evolution (EnMODE)
- Energy-aware Non-Dominated Sorting Genetic Algorithm (EnNSGA-II)

## Features

- EnMODE generates only one solution based on the objective weights
- EnNSGA-II generates a set of solutions called Pareto-front
 

## Documntation
The source codes belogs to the following paper: 
**Seyed Jalaleddin Mousavirad and Luís A. Alexandre, Energy-Aware JPEG Image Compression: A Multi-Objective Approach**

## How to cite 

@article{JPEG_MO,
	title={Energy-Aware JPEG Image Compression: A Multi-Objective Approach},
	author={Mousavirad, Seyed Jalaleddin and A. Alexandre, Luís},
	journal={arXiv preprint },
	year={2022}
}

