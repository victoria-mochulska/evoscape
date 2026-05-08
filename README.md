<img src="figures/evoscape.png" alt="Logo" width="300"/>

A simulation framework for constructing and optimizing epigenetic landscape models 

## 📍 Features
* Parametrized, interpretable landscapes built with Waddington-like valleys
* Modular construction using local flow elements in 2D
* Flexible topography and topology with minimal constraints
* Optimization algorithm inspired by biological evolution 


## 🌀 Landscape construction
<p align="center">
  <img src="figures/Figure1_intro.png" alt="Project Logo" width="700"/>
</p>


## 📁 Structure 

<pre>

evoscape/
├── modules/
│   └── module_class.py                 # Module definitions
├── landscapes/
│   ├── landscape_class.py              # Core landscape definition
│   ├── landscape_dataset_fitness.py    # Landscape for fitting a timelapse dataset
│   └── landscape_segmentation.py       # Landscape for modelling tissue segmentation
├── population_class.py                 # Evolution in a population of landscapes
├── morphogen_regimes.py                # Temporal dependencies of parameters
├── helper_functions.py
├── module_helper_functions.py
└── landscape_visuals.py

- <b>examples</b>/: Example usage of Evoscape, quick optimization runs 
- scripts/: Codes for parallelized optimization, multiple runs
- notebooks/: Jupyter notebooks used for analysis and figures
</pre>

### Getting started
Check out `examples` notebooks


### 🚧 In development
* Interactive simulation

## 📃 Related publication

 V. Mochulska, P. François (2025). **Generative epigenetic landscapes map the topology and
topography of cell fates**. PNAS [Paper link](https://www.pnas.org/doi/10.1073/pnas.2514508122) |
[bioRxiv link](https://www.biorxiv.org/content/10.1101/2025.06.09.658705v2)

<details>
<summary>Citation</summary>

```bibtex
@article{
doi:10.1073/pnas.2514508122,
author = {Victoria Mochulska  and Paul François },
title = {Generative epigenetic landscapes map the topology and topography of cell fates},
journal = {Proceedings of the National Academy of Sciences},
volume = {122},
number = {50},
pages = {e2514508122},
year = {2025},
doi = {10.1073/pnas.2514508122},
URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2514508122},
eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2514508122}
}
```
</details>

## References

**Scientific colormaps**: Crameri, F. (2018), Scientific colour maps, doi:10.5281/
zenodo.1243862

## Contact

`victoria dot mochulska at mail dot mcgill dot ca` 
