# nonergodicGMM
This repository implements the core code base for the development and dynamic update of a non-ergodic GMM. 


## Installation

### Dependencies

The complete set of tools requires the following packages:

```
openquake >= 3.19
numexpr
Rtree
geopandas
obspy
```

To run the INLA regression tools we also recommend installation of `R >= 4.3.0`, including the `r-inla` package


### Installation via `pip`

**Given the dependency stack of openquake and obspy we _strongly_ recommend installing the tools within a virtual environment. For further instructions see: <https://docs.python.org/3/library/venv.html>**

Using `git` first clone the repository to a local folder:

```
git clone https://github.com/g-weatherill/nonergodicGMM.git
```

Then run:

```
pip install nonergodicGMM/
```

To make changes to the code we recommend installing the code as editable:

```
pip install -e nonergodicGMM/
```


## Credits

Elements of the process for developing and running non-ergodic GMMs are adapted from descriptions, formats and code examples provides by Lavrentiadis et al. (2022).

```
Lavrentiadis, G., Kuehn, N., Bozorgnia, Y., Seylabi, E., Meng, X., Goulet, C.,
and Kottke, A. (2024) Non-ergodic Methodology and Modelling Tools. Natural Hazards
Risk and Resiliency Center Report GIRS-2022-04, University of California,
Los Angeles, August 2022, DOI: 10.34948/N35P4Z,
https://www.risksciences.ucla.edu/girs-reports/2022/04
```
