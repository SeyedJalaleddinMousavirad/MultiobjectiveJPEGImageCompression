## Multi-objective JPEG Image Compression (MOJPEG)

This repository includes the source codes for the folloiwng paper:

**Seyed Jalaleddin Mousavirad and Luís A. Alexandre, Energy-Aware JPEG Image Compression: A Multi-Objective Approach**

MOJPEG is an approch for Multiobjective JPEG Image Compression, which uses PYMOO framwork in Python to find proper quantisation tables(QT). Two general goals is to maximise the image quality and minimise image file size.
Here, we employed two general multi-objective metaheuristic approaches: scalarisation and Pareto-based. Therefore, here, we provide two algorithms, one scalarisation method and one Pareto-based method.

- Energy-aware Multi-objective Differential Evolution (EnMODE)
- Energy-aware Multi-objective Genetic Algorithm (EnMOGA)
- Energy-aware Multi-objective Particle Swarm Optimisation (EnMOPSO)
- Energy-aware Multi-objective Evolutionary Startegy (EnMOES)
- Energy-aware Multi-objective Particle Search (EnMOPS)
- Energy-aware Non-Dominated Sorting Genetic Algorithm (EnNSGA-II)
- Energy-aware Reference-based Non-Dominated Sorting Genetic Algorithm (EnNSGA-III)

## Features

- Scalarisation methods, EnMODE, EnMOGA,‌ENMOPSO,ENMOES, and ENMOPS, generate only one solution based on the objective weights
- Pareto-based approches, EnNSGA-II and EnNSGA-III, generate a set of solutions called Pareto-front
 

## Documntation
The source codes belogs to the following paper: 

**Seyed Jalaleddin Mousavirad and Luís A. Alexandre, Energy-Aware JPEG Image Compression: A Multi-Objective Approach**

## Installation

## How to cite 

@article{JPEG_MO,
	title={Energy-Aware JPEG Image Compression: A Multi-Objective Approach},
	author={Mousavirad, Seyed Jalaleddin and A. Alexandre, Luís},
	journal={arXiv preprint },
	year={2022}
}

