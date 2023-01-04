# FRAME
## FRactionation And Mixing Evaluation

Welcome to FRAME! A friendly graphical interface for calculation of isotope mixing.

[Article at PLOS ONE](https://doi.org/10.1371/journal.pone.0277204)

## About

FRAME is a Bayesian stable isotope mixing model for **sources partitioning AND simultaneous estimation of fractionation progress** based on the stable isotope composition of sources/substrates and mixture/products. The underlying mathematical algorithm is the **Markov-Chain Monte Carlo** implemented in python. It features an open mathematical design allowing for the **implementation of custom additional processes** that alternate the characteristics of the final mixture and its application for a broad range of studies. It comes with a **friendly graphical interface** too!

## Install

Latest release:
 - **v1.0** [Download [EXE]](https://github.com/malewick/frame/releases/tag/v1.0)

The python source code and instructions for developers are given on the [github page](https://github.com/malewick/frame).

## User Guide

The program works in 5 simple steps:

| Step  | Screenshot |
| ------------- | ------------- |
| 1. Load data on your measured samples  | [<img width="500" alt="scrrenshot loading data" src="https://user-images.githubusercontent.com/24914567/204164202-4426b6dc-0f3d-4f73-81bf-cbb5fe2ce698.png">](https://user-images.githubusercontent.com/24914567/204164202-4426b6dc-0f3d-4f73-81bf-cbb5fe2ce698.png) |
| 2. Load isotopic signatures of considered sources | [<img width="500" alt="screanshot loading sources" src="https://user-images.githubusercontent.com/24914567/204164209-0d958310-640f-4eb0-a50a-664395ad0a7a.png">](https://user-images.githubusercontent.com/24914567/204164209-0d958310-640f-4eb0-a50a-664395ad0a7a.png) |
| 3. (Optional: Load details on the fractionation process)  | [<img width="500" alt="screenshot loading fractionation" src="https://user-images.githubusercontent.com/24914567/204164214-d3dd12e6-0f70-4ec7-a802-e87e64018f40.png">](https://user-images.githubusercontent.com/24914567/204164214-d3dd12e6-0f70-4ec7-a802-e87e64018f40.png)  |
| 4. Verify your loaded data  | [<img width="500" alt="screenshot data plotted" src="https://user-images.githubusercontent.com/24914567/204164250-67b016b9-c780-4cad-ad26-68a0aa868a9b.png">](https://user-images.githubusercontent.com/24914567/204164250-67b016b9-c780-4cad-ad26-68a0aa868a9b.png)  |
| 5. Run the model! | [<img width="260" alt="screenshot run model" src="https://user-images.githubusercontent.com/24914567/204164219-64025d4e-db2d-45cf-b668-3ae0bc5d4816.png">](https://user-images.githubusercontent.com/24914567/204164219-64025d4e-db2d-45cf-b668-3ae0bc5d4816.png)  |

Voila! The simulation is running!

[<img width="800" alt="screenshot model running" src="https://user-images.githubusercontent.com/24914567/204164285-dd67e181-00f9-4abe-9c34-34ea7ffe56fe.png">](https://user-images.githubusercontent.com/24914567/204164285-dd67e181-00f9-4abe-9c34-34ea7ffe56fe.png)

**A more detailed user guide can be found under [this link](user_guide.md).**

A much more detailed scientific description, including an explanation of some real-world use cases is published here: [link to the paper -- coming soon!].

The example input data are included in the installation and for the real-world examples see: [real-world-examples](https://github.com/malewick/frame/tree/main/input/real%20world%20examples)



## Discussion and bugs report

Our mailing list is at https://groups.google.com/g/frame-isotopes. In case of any questions regarding the program (underlying algorithms, usage, etc.) please write to us there.

Our issue tracker is at https://github.com/malewick/frame/issues. Please report any bugs that you find. 


## Atribution
Please cite FRAME in publications using:

Lewicki MP, Lewicka-Szczebak D, Skrzypek G (2022) _FRAME—Monte Carlo model for evaluation of the stable isotope mixing and fractionation_.
PLoS ONE 17(11): e0277204. https://doi.org/10.1371/journal.pone.0277204


## License, Distribution and Disclaimer
FRAME is GPL3 licensed (see the [LICENSE file](https://github.com/malewick/frame/blob/main/LICENSE) for details). Please cite our work when using it in your work and also consider documenting and contributing all your changes back, so that we can incorporate it and all of us will benefit in the end.

FRAME is provided free of charge and is distributed in the hope that it will be useful. This software is provided "as is" and without warranty. Use at your own risk. The authors hereby disclaim any liability for the use of this freeware software. If you do not accept these conditions please do not download or use FRAME.

## Authors
 - **Maciej Lewicki**, malewick[at]cern.ch, https://malewick.web.cern.ch/
 - **Dominika Lewicka-Szczebak**, dominika.lewicka-szczebak[at]uwr.edu.pl 
 - **Grzegorz Skrzypek**, grzegorz.skrzypek[at]uwa.edu.au, http://www.gskrzypek.com/
