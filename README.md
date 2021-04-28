# FRAME
## FRactionation And Mixing Evaluation of isotopes.

Markov-chain Monte Carlo engine for calculation of isotope mixinng wrapped in a simple Qt user interface.

See the [webpage](https://malewick.github.io/frame/) for a user guide, information of newest releases and more.

See the paper for a more detailed scientific description: [link to the paper]

In the `src` directory:
- `FRAME.py` is the main script producing the GUI.
- `NDimModel.py` contains the implementation of the model and statstical calculations.
- `TimeSeriesPlot.py`, `PathPlot.py`, `CorrelationPlot.py` are the classes with mpl plots implementation.
- `mxml.py` is a simple script for running the model without the GUI, useful for batch (distributed) computing.

`requirements.txt` lists all the necessary packages.

`input` directory contains useful examples for running the code 
