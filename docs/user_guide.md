# FRAME
## FRactionation And Mixing Evaluation

Welcome to FRAME! A friendly graphical interface for calculation of isotope mixing.

## Start up

Just double-click the executable. You will usually need to wait for couple seconds, first the system console will appear and then the GUI will launch.
Waiting times might differ for different machines.

## Main Interface

![screenshot0](https://user-images.githubusercontent.com/24914567/118247339-5bef2400-b4a3-11eb-9928-9b9d0d2f8916.png)

### Loading input data

In order to launch calculation you need to load input data on: measured samples, isotopic signatures of sources and optionally a data on other parameters included in the model.

1. **Load samples**  
The measured isotopic signatures of the samples should be introduced into the model by defining the following columns:
- *group*- the following numbers of samples groups representing common samples class or series
- *label* - identification of individual samples
- isotopic signatures - in two columns for 2D model and three columns for 3D model - with name of isotopic signature as the heading (e.g. δ<sup>18</sup>O) and measured isotopic δ values as entries for each sample
- standard deviation of each isotopic measurement in two columns for 2D model and three columns for 3D model - with heading as sigma(isotopic signature), e.g. sigma(δ 18 O), and the values of measurement uncertainties as entries for each sample

2. **Load sources**
The isotopic characteristic of the sources should be introduced into the model by defining the following columns:
- *source* – name or abbreviation of the source
- isotopic signatures in two columns for 2D model and three columns for 3D model - with name of isotopic signature as the heading, identical to the headings defined in samples file (e.g.δ 18 O) and mean isotopic δ values as entries for each source. In case of range of equally probable values representing the source the mean value = (range max + range min)/2.
- delta (isotopic signature) - e.g. delta(δ 18 O) – the spread of the isotopic range of the particular source, i.e., for delta = (range max + range min)/2
the range is defined as mean value ± delta. All the values within this range have equal probability. In case of one well determined mean value of the highest probability you should insert 0 as delta value.
- sigma (isotopic signature) – e.g. sigma(δ 18 O) - standard deviation representing the analytical uncertainty of the isotopic analysis of the source values – the range ± sigma is added to the mean value ± delta range as a margin with Gaussian likelihood (see Figure 8). By inserting 0 as sigma you can omit this uncertainty.

3. Optional: **Load auxiliary data**
With this file the additional parameter associated with isotopic fractionation can be added, by defining the following rows:
- row 1 - define the model equation including auxiliary data, where: M0[i] represents the final isotopic mixture of all sources before fractionation capital letters [i] represent the parameters, e.g. the isotopic fractionation factors; letter "r" represents the unknown quantity, e.g. the residual fraction (use r_1 , r_2 in case of more unknowns, see practical examples below)
- row 2 - define the columns for entering data of the defined parameters: ’name’ - indicate which parameter (capital letter); indicate the following isotopic values which are used by the model with identical headings as used in samples and sources file; delta (isotopic signature) - analogically as in source file defines the spread of the range; sigma (isotopic signature) - analogically as in source file defines the analytical uncertainty.
- the following rows - for the entries of values for the above defined parameters

### Examples of model equations

Examples of fractionation equations and model entries:
- Open system fractionation
```M = M0[i] + E[i]*r```
where `M` stands for the final isotopic signature,` M0` stands for the isotopic signature of initial mixture before fractionation:
```M[0]=f[0]*S[i][0]+f[1]*S[i][1]+f[2]*S[i][2]...```
- where `f` is the fraction of each source and `S[i]` is the characteristic isotopic signature of each source, `E[i]` stands for isotopic fractionation factor
and `r` for the residual unreacted fraction.  
An example of this fractionation is included in the case study described in Sect. 4.1 and in the example file NO3_frac.csv
- Closed system (Rayleigh-type) fractionation
```M = M0[i] + E[i]*log(r)```
where `M` stands for the final isotopic signature, `M0` stands for the isotopic signature of initial mixture before fractionation:
```M[0]=f[0]*S[i][0]+f[1]*S[i][1]+f[2]*S[i][2]...```
- where `f` is the fraction of each source and `i` is the characteristic isotopic signature of each source, `E[i]` stands for isotopic fractionation factor and
`r` for the residual unreacted fraction. Note that the natural logarithm is denoted with log in the model language.  
An example of this fractionation is included in the case study described in Sect. 4.3 and in the example file N2O_frac.csv
- Equilibrium fractionation
```M = M0[i]*(1-r*A[i]) + D[i]* r * A[i]```
where `M` stands for the final isotopic signature, `M0` stands for the isotopic signature of initial mixture before fractionation:
```M[0]=f[0]*S[i][0]+f[1]*S[i][1]+f[2]*S[i][2]...```
where `f` is the fraction of each source and `i` is the characteristic isotopic signature of each source, `D[i]` stands for isotopic signature after equilibration and `r` for the equilibrated fraction. The additional parameter `A[i]` can be defined for the isotope undergoing equilibration as 1 and for the isotope which is not undergoing equilibration as 0, or any other values depending on the equilibration ratio between various isotopes.  
An example of this fractionation is included in the case study described in Sect. 4.3 and in the example file NO2_frac.csv

You can also save your current configuration into xml file, which can later be conveniently loaded to resume work (`File→Save`, `File→Load`).

Once you run the model you will see the output console informing you about the state of the calculation and three additional plotting canvases will
appear.
- Figure 1. Time-series of entries accepted by the Metropolis-Hastings algortihm, that build the Markov chain. The burnout period is marked with a dashed lined.
- Figure 2. On the diagonal there are histograms showing distributions of evaluated variables. Panels above the diagonal show the correlation between the variables and panels below the diagonal show the same correlation, but evaluated as a single number.
- Figure 3. For each accepted set of variables the isotope mixture is evaluated and plotted as a path.

These plots serve the purpose of on-line quality assessment and it is advised to always begin analyses with running a couple of test samples to see if everything
works as expected. However, when running the analysis for multiple samples, it is advised to turn off the on-line plotting, as the computation time is then
greatly reduced.

Also, when the model is run the FRAME interface switches to a new tab, which contains the output console and the progress bar to inform you about
the status of the computation

![screenshot1](https://user-images.githubusercontent.com/24914567/118250026-52b38680-b4a6-11eb-9c6a-681721fce6aa.png)
