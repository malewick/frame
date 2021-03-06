# FRAME
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
## FRactionation And Mixing Evaluation of isotopes.

Markov-chain Monte Carlo engine for calculation of isotope mixinng wrapped in a simple Qt user interface.

See the [webpage](https://malewick.github.io/frame/) for a user guide, information of newest releases and more.

## Install
### Executables
- **Windows**: see the zipped executables in [releases](https://github.com/malewick/frame/releases/) (only Windows 10 compatible for now)
- **Linux** and **OSX**: no exectuables were prepared yet. Let us know if you wish to use our software on these platforms or run the code from sources as described below.

### Install from source
To get the sources from git do

``
$ git clone https://github.com/malewick/frame.git:
``

Install the packages listed in `requirements.txt` (either with conda or pip). Then run:

``
python FRAME.py
``

Alternatively you can run the code without the graphical interface with (see the source for details and `input` directory for some sample input files):

``
python run_batch.py
``

## Docs and user guide

The user guide can be found under [this link](https://malewick.github.io/frame/)

A more detailed scientific description is published here: [link to the paper]

## Discussion and bugs report

Our mailing list is at https://groups.google.com/g/frame-isotopes. You can ask questions about the underlying algorithms, usage, etc.

Our issue tracker is at https://github.com/malewick/frame/issues. Please report any bugs that you find. 

## Code
In the `src` directory:
- `FRAME.py` is the main script producing the GUI.
- `NDimModel.py` contains the implementation of the model and statstical calculations.
- `TimeSeriesPlot.py`, `PathPlot.py`, `CorrelationPlot.py` are the classes with mpl plots implementation.

`requirements.txt` lists all the required packages and versions (PySide2, pandas, numpy, sympy, matplotlib).

## Attribution
Please cite the following paper when using this code:

- [to be supplemented]

It is expected that changes to the code are documented.

*Copyright (C) 2020-2021  Maciej P. Lewicki*
