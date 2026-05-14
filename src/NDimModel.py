"""@package fracMCMC
Markov chain Monte Carlo for modeling the mixing of stable isotopes.
 
Markov chain Monte Carlo for modeling the mixing of stable isotopes.
v1.0
Feb 2021
author: Maciej P. Lewicki
malewick@cern.ch
"""

import pandas as pd
import numpy as np
from numpy import *
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

import time

import xml.etree.ElementTree as ET

import sympy

import TimeSeriesPlot
import PathPlot
import CorrelationPlot
import ResultsPlot

from scipy.stats import norm, uniform
from scipy.signal import fftconvolve


def splitall(s):
	for delimiter in [';',',','\t']:
		if delimiter in s :
			s=s.split(delimiter)
			if type(s) is not str:
				s = list(filter(lambda x : x!=delimiter, s))
				s = list(filter(None, s))
	if type(s) is str:
		return [s]
	else :
		return s

class Plotter:

	def __init__(self):
		self.switch_1=True
		self.switch_2=True
		self.switch_3=False


	def initialize_plots(self,model,group,plot3_style,show=True):
		# ion for live updates
		if show:
			plt.ion()
		else:
			pass

		if self.switch_1:
			# figure 1 -- the chains
			self.plot1 = TimeSeriesPlot.TimeSeriesPlot(model)
			# figure 2 -- the path on the isotope map
			self.plot2 = PathPlot.PathPlot(model,group)

		if self.switch_2 :
			# figure 3 -- histograms and correaltion of Markov chain variables`
			self.plot3 = CorrelationPlot.CorrelationPlot(model)
			self.plot3_style = plot3_style
			print("using correlation plot style: ", plot3_style)
			self.plot3.plot_type = plot3_style

		# ----------------------------------


	def draw_at_checkpoint(self,model) :

		if self.switch_2 and len(model.xd1[0])>1 :
			self.plot3.update(model)

	def draw_at_metropolis(self,model,M,r,f) :

		if self.switch_1 :
			self.plot1.update(model,M,r,f)
			self.plot2.update(model,M)

	def finished_plotting(self,model,group) :

		if self.switch_1:
			self.plot1.save(model.output_dir+'/'+model.dataset_name+"_"+"group_"+str(group)+"_",model.myfmt)
			self.plot2.save(model.output_dir+'/'+model.dataset_name+"_"+"group_"+str(group)+"_",model.myfmt)
		if self.switch_2:
			self.plot3.save(model.output_dir+'/'+model.dataset_name+"_"+"group_"+str(group)+"_",model.myfmt)

	def draw_final_dists(self,xdata,model) :

		if not self.switch_3:
			return

		fig, ax = plt.subplots(model.nvariables, 2, sharex=True)

		print(model.var_list)

		for i,var in enumerate(model.var_list) :

			print(var)
			xhist, xbins = np.histogram(xdata[i], bins=100, range=(0,1))
			bincentres = [(xbins[i]+xbins[i+1])/2. for i in range(len(xbins)-1)]
			ax[i][0].step(bincentres,xhist,where='mid')
			xhist = xhist.astype(float)
			xcum = np.cumsum(xhist)
			xcum /= xcum[-1]
			ax[i][1].step(bincentres,xcum,where='mid')
			ax[i][0].set_title(var)

		plt.show(block=True)

	def draw_only_at_end(self,model,group,plot3_style):
		self.initialize_plots(model,group,plot3_style,False)
		self.draw_at_checkpoint(model)
		self.plot1.xdata = list(range(0,len(model.xb1[0])))
		self.plot1.ydata = model.xb1[1:].copy()
		self.plot1.ydata.append(model.xb1[0])
		self.plot1.update_graph(model)
		self.plot2.draw_graph(model)
		self.finished_plotting(model,group)


class Model:

	def __init__(self):

		# online plotting, True by default
		self.plotting_switch=True
		self.threading_safe=False

		self.set_iterations()
		
		# 1 - verbose, 2 - limited, 3 - only important
		self.printlevel=1
		self.nauxvars=1
		self.aux_toggle=False
		self.aux_pars=[]
		self.aux_vars=[]
		self.model_derivatives=[]
		self.aux=[]
		self.aux_stdev=[]

		self.erf_toggle = True

		# Auxiliary variable prior settings: maps var_name -> (prior_type, r_min, r_max)
		# Supported prior_type values: 'uniform', 'loguniform'
		# Defaults to uniform[0, 1] if not explicitly set.
		self.aux_prior = {}

		self.abort = False
		self.latex = False

		self.chain_counter=0
		self.chain_checkpoint_every=20
		self.burnout_chain_len=0

		self.axes_labels=[]
		self.default_delimiter=','

		# was the 'set_up_data' method called?
		self.initialized=False

		# 0 - just the regular output
		# 1 - technical and mathematical details
		# 2 - coding debug
		self.verbosity=1
	
		self.dataset_name=""
		self.myfilename=""
		self.myfmt=""

		# I didn't want to have plotting-related variables here, but I cannot think of anything simpler for now:
		self.plot3_style='hist2d'

	def reset(self) :
		# was the 'set_up_data' method called?
		self.initialized=False
		self.set_up_data()
		self.abort=False


	def load_from_fielnames(self, data_file, sources_file, aux_file=None):
		self.load_measurements(data_file)
		self.load_sources(sources_file)
		if aux_file is not None and aux_file != "":
			self.load_aux(aux_file)

		self.set_up_data()

	def save_to_xml(self, xml_file):

		# create the file structure
		data = ET.Element('data')
		items = ET.SubElement(data, 'items')
		item1 = ET.SubElement(items, 'item')
		item2 = ET.SubElement(items, 'item')
		item1.set('name','data_file')
		item2.set('name','sources_file')
		item1.text = self.data_file
		item2.text = self.sources_file
		if self.aux_file != None:
			item3 = ET.SubElement(items, 'item')
			item3.set('name','aux_file')
			item3.text = self.aux_file

		# create a new XML file with the results
		mydata = ET.tostring(data)
		myfile = open(xml_file, "w")
		myfile.write(mydata)

	def load_from_xml(self, xml_file):

		tree = ET.parse(xml_file)
		root = tree.getroot()
		dict_from_xml={}
		for elem in root:
			for subelem in elem:
				print(subelem.attrib, subelem.text)
				dict_from_xml[subelem.attrib['name']] = subelem.text
		print(dict_from_xml)

		self.load_measurements(dict_from_xml['data_file'])
		self.load_sources(dict_from_xml['sources_file'])
		if 'aux_file' in dict_from_xml.keys():
			self.load_aux(dict_from_xml['aux_file'])

		self.set_up_data()


	def load_measurements(self, data_file):

		# Create measurements DataFrame
		self.data_file = data_file
		self.df = pd.read_csv(data_file, skiprows = 0, delimiter="[,\t]", engine='python')
		new_col = self.df.index
		self.df.insert(loc=0, column='sample_id', value=new_col)
		print("Measurements dataframe:")
		print(self.df)

		self.isotopes_list=[]
		for var in [*self.df.columns]:
			if var!="group" and var!="label" and "stdev" not in var and var!="sample_id":
				self.isotopes_list.append(var)
		print("isotopes_list: ",self.isotopes_list)
		print()

	def load_sources(self, sources_file):

		# Create sources DataFrame 
		self.sources_file=sources_file
		self.df_sources = pd.read_csv(sources_file, skiprows = 0, delimiter="[,\t]", engine='python')
		print("Sources dataframe:")
		print(self.df_sources)

		self.sources_list=[]
		for var in self.df_sources["source"].unique():
			self.sources_list.append(var)
		print("sources list:", self.sources_list)
		print()

		spread_sum=0
		for iso in self.isotopes_list:
			for spread in self.df_sources["spread("+iso+")"]:
				spread_sum += spread
		if spread_sum == 0:
			print("All source spreads are zero: likelihood will use the pure-Gaussian path for all isotopes.")
			self.erf_toggle = False
		else :
			print("Nonzero source spreads detected: likelihood will use numerical convolution per isotope.")
			


	def load_aux(self, aux_file):

		self.aux_file=aux_file
		# Create auxiliary parameter DataFrame
		print("--- Aux file import ---")

		with open(aux_file) as f:
			content = f.readlines()
		content = [x.strip() for x in content]

		print(content[0])
		model_equation = splitall(content[0])
		model_equation=model_equation[0]
		print("model_equation: ",model_equation)

		self.df_aux = pd.read_csv(aux_file, skiprows = 1, delimiter='[,\t]', engine='python')
		print("Auxiliary parameter DataFrame")
		print(self.df_aux)
		print()

		self.model_definition=model_equation
		self.aux_pars=list(self.df_aux['name'].unique())
		self.aux_vars=list(set(re.findall(r'r\d',self.model_definition)))
		if len(self.aux_vars)==0:
			self.aux_vars=list(set(re.findall(r'r',self.model_definition)))
			if len(self.aux_vars)>1:
				print("ERROR: something's wrong with declaration of auxiliary variables in the model.")

		self.aux_toggle=True

		print("auxiliary parameters list:", self.aux_pars)
		print("auxiliary variables list:", self.aux_vars)



	def set_outdir(self, out_dir=".") :

		# Output directory and files
		self.output_dir=out_dir+"/"
		datafname=re.sub('.*/', '', re.sub('.csv', '', self.data_file))
		sourcesfname=re.sub('.*/', '', re.sub('.csv', '', self.sources_file))
		if self.aux_toggle :
			auxfname=re.sub('.*/', '', re.sub('.csv', '', self.aux_file))
			self.dataset_name=datafname+"_"+sourcesfname+"_"+auxfname
		else :
			self.dataset_name=datafname+"_"+sourcesfname
		if self.myfilename!="":
			self.dataset_name=self.myfilename
		self.output_dir=self.output_dir+self.dataset_name
		Path(self.output_dir).mkdir(parents=True, exist_ok=True)
		self.output=self.output_dir+'/'+self.dataset_name+'_results.csv'
		print("Output file: ")
		print(self.output)
		print()

	def set_iterations(self, niter=1e7, burnout=100, max_chain_entries=500) :

		# number of iterations, burnout, desired Markov chain length
		self.niter=int(niter)
		self.burnout=int(burnout)
		self.max_chain_entries=int(max_chain_entries)

	def set_aux_prior(self, var_name, prior_type='uniform', r_min=0.0, r_max=1.0) :
		"""Configure the sampling prior for an auxiliary variable.

		Parameters
		----------
		var_name : str
		    Name of the auxiliary variable as it appears in the model equation
		    (e.g. 'r', 'r1', 'r2').
		prior_type : str
		    Sampling distribution. Supported values:
		    - 'uniform'    — draws from Uniform[r_min, r_max]  (default)
		    - 'loguniform' — draws from LogUniform[r_min, r_max],
		                     i.e. log(r) ~ Uniform[log(r_min), log(r_max)].
		                     Requires r_min > 0 and r_max > 0.
		r_min : float
		    Lower bound of the prior range (default 0.0).
		r_max : float
		    Upper bound of the prior range (default 1.0).

		Examples
		--------
		>>> model.set_aux_prior('r', prior_type='uniform', r_min=0.0, r_max=1.0)
		>>> model.set_aux_prior('r', prior_type='loguniform', r_min=0.001, r_max=1.0)
		"""
		supported = ('uniform', 'loguniform')
		if prior_type not in supported:
			raise ValueError(
				f"Unknown prior_type '{prior_type}'. Supported: {supported}"
			)
		r_min = float(r_min)
		r_max = float(r_max)
		if prior_type == 'loguniform' and (r_min <= 0 or r_max <= 0):
			raise ValueError(
				f"loguniform prior requires r_min > 0 and r_max > 0 "
				f"(got r_min={r_min}, r_max={r_max})"
			)
		if r_min >= r_max:
			raise ValueError(
				f"r_min must be strictly less than r_max "
				f"(got r_min={r_min}, r_max={r_max})"
			)
		self.aux_prior[var_name] = (prior_type, r_min, r_max)

	def _sample_aux_prior(self, var_name) :
		"""Draw one sample from the configured prior for *var_name*."""
		prior_type, r_min, r_max = self.aux_prior.get(var_name, ('uniform', 0.0, 1.0))
		if prior_type == 'loguniform':
			return float(np.exp(np.random.uniform(np.log(r_min), np.log(r_max))))
		else:
			return float(np.random.uniform(r_min, r_max))

	def set_up_data(self) :

		if self.initialized==True:
			return
		else:
			self.initialized=True

		if self.aux_toggle :
			model_definition_preregex = self.model_definition
			model_aux_pars_set = set(re.findall('[A-Z]',model_definition_preregex))
			for char in ['S','M']:
				if char in model_aux_pars_set: model_aux_pars_set.remove(char)
			if model_aux_pars_set != set(self.aux_pars) :
				print("ERROR: Something's wrong with aux par definitions.")

			for i, auxpar in enumerate(self.aux_pars):
				self.model_definition = re.sub(r'\b'+str(auxpar)+r'\b\[i\]','aux[i]['+str(i)+']',self.model_definition)
			for i, auxvar in enumerate(self.aux_vars):
				self.model_definition = re.sub(r'\b'+str(auxvar)+r'\b','r['+str(i)+']',self.model_definition)

			strM0=''
			for i in range(len(self.sources_list)):
				strM0+='S[i]['+str(i)+']*f['+str(i)+']+'
			strM0 = strM0[:-1]
			self.model_definition = re.sub(r'M0\[i\]',strM0,self.model_definition)
			self.model_definition = self.model_definition.lstrip('\ufeff')

			self.sym_model_definition = re.sub(r'\[(\d*)\]',r'\1',self.model_definition)
			self.sym_model_definition = re.sub(r'\[i\]','',self.sym_model_definition)

			self.sym_model_derivatives=[]
			self.model_derivatives.clear()
			str_symbols=''
			for i in range(len(self.sources_list)):
				str_symbols+='S'+str(i)+' '
			for i in range(len(self.aux_pars)):
				str_symbols+='aux'+str(i)+' '
			str_symbols = str_symbols[:-1]
			sym_vars = sympy.symbols(str_symbols)
			for var in sym_vars:
				dvardM = str(sympy.diff(self.sym_model_definition,var))
				self.sym_model_derivatives.append(dvardM)
				dvardM = re.sub(r'f(\d*)',r'f[\1]',dvardM)
				dvardM = re.sub(r'r(\d*)',r'r[\1]',dvardM)
				dvardM = re.sub(r'aux(\d*)',r'aux[i][\1]',dvardM)
				dvardM = re.sub(r'S(\d*)',r'S[i][\1]',dvardM)
				self.model_derivatives.append(dvardM)

			print()
			print("Model equation")
			print("Before regex:\t",model_definition_preregex)
			print("After regex:\t",self.model_definition)
			print("Symbolic:\t",self.sym_model_definition)
			print()
			print("and derivatives")
			print("Symbolic:\t",self.sym_model_derivatives)
			print("After regex:\t",self.model_derivatives)
			print()

		#model_pars = symbols()

		self.par_list = self.sources_list + self.aux_pars
		self.var_list = self.sources_list + self.aux_vars

		self.nsources=len(self.sources_list)
		self.nisotopes=len(self.isotopes_list)
		self.nauxpars=len(self.aux_pars)
		self.nauxvars=len(self.aux_vars)
		self.nparameters=len(self.par_list)
		self.nvariables=self.nsources+self.nauxvars

		self.sources=np.zeros(shape=(self.nisotopes,self.nsources),dtype='double')
		self.sources_stdev=np.zeros(shape=(self.nisotopes,self.nsources),dtype='double')
		self.sources_spread=np.zeros(shape=(self.nisotopes,self.nsources),dtype='double')
		for i,iso in enumerate(self.isotopes_list):
			for j,src in enumerate(self.sources_list):
				self.sources[i][j]=self.df_sources.loc[ (self.df_sources['source']==src) ].iloc[0][iso]
				self.sources_stdev[i][j]=self.df_sources.loc[ (self.df_sources['source']==src) ].iloc[0]["stdev("+iso+")"]
				self.sources_spread[i][j]=self.df_sources.loc[ (self.df_sources['source']==src) ].iloc[0]["spread("+iso+")"]

		print("self.sources")
		print(self.sources)
		print("self.sources_stdev")
		print(self.sources_stdev)
		print("self.sources_spread")
		print(self.sources_spread)
		print()
				
		if self.aux_toggle:
			print('Numer of auxiliary parameters:',len(self.aux_pars))
			print(self.aux_pars)

			print('Numer of auxiliary variables:',self.nauxvars)

			self.aux=np.zeros(shape=(self.nisotopes,self.nauxpars),dtype='double')
			self.aux_stdev=np.zeros(shape=(self.nisotopes,self.nauxpars),dtype='double')
			self.aux_spread=np.zeros(shape=(self.nisotopes,self.nauxpars),dtype='double')
			for i,iso in enumerate(self.isotopes_list):
				for j,nm in enumerate(self.aux_pars):
					self.aux[i][j]=self.df_aux.loc[ (self.df_aux['name']==nm)].iloc[0][iso]
					self.aux_stdev[i][j]=self.df_aux.loc[ (self.df_aux['name']==nm) ].iloc[0]["stdev("+iso+")"]
					self.aux_spread[i][j]=self.df_aux.loc[ (self.df_aux['name']==nm) ].iloc[0]["spread("+iso+")"]

			print("self.aux:", self.aux)
			print("self.aux_stdev:", self.aux_stdev)
			print("self.aux_spread:", self.aux_spread)
			print()

		if self.model_derivatives==[]:
			for i in range(self.nsources):
				self.model_derivatives.append("f["+str(i)+"]")

		self.mapPar={}
		self.stdevPar={}
		self.spreadPar={}
		self.dMdPar={}
		print("----")
		print(self.sources_list)
		print(self.model_derivatives)
		for j in range(self.nsources) :
			self.mapPar[self.sources_list[j]] = [self.sources[i][j] for i in range(self.nisotopes)]
			self.stdevPar[self.sources_list[j]] = [self.sources_stdev[i][j] for i in range(self.nisotopes)]
			self.spreadPar[self.sources_list[j]] = [self.sources_spread[i][j] for i in range(self.nisotopes)]
			self.dMdPar[self.sources_list[j]] = self.model_derivatives[j]
		for j in range(self.nauxpars) :
			self.mapPar[self.aux_pars[j]] = [self.aux[i][j] for i in range(self.nisotopes)]
			self.stdevPar[self.aux_pars[j]] = [self.aux_stdev[i][j] for i in range(self.nisotopes)]
			self.spreadPar[self.aux_pars[j]] = [self.aux_spread[i][j] for i in range(self.nisotopes)]
			self.dMdPar[self.aux_pars[j]] = self.model_derivatives[self.nsources + j]

		# Pre-compile derivative expressions so the hot MCMC loop avoids
		# re-parsing strings on every iteration.
		self.dMdPar_code = {par: compile(expr, "<dMdPar>", "eval") for par, expr in self.dMdPar.items()}
		if self.aux_toggle:
			self.model_definition_code = compile(self.model_definition, "<model_definition>", "eval")

		print("self.mapPar: ",self.mapPar)
		print("self.stdevPar: ",self.stdevPar)
		print("self.dMdPar: ",self.dMdPar)
		print()

		# to store everything after burnout
		# xd1 stores variables: f_1, ..., r_1,...
		# xd2 stores mu_i
		self.xd1 = [ [] for i in range(self.nvariables)]
		self.xd2 = [ [] for i in range(self.nisotopes)]

		# to store everything, including burnout
		self.xb1 = [ [] for i in range(self.nvariables)]
		self.xb2 = [ [] for i in range(self.nisotopes)]

		if len(self.axes_labels)==0:
			self.axes_labels = self.isotopes_list.copy()


	def sim_finished(self,group,f_out) :
		xdata=self.xd1
		print("Simulation has finished.")

		print("Output directory:", self.data_file)

		print("Results:")

		if self.plotting_switch :
			self.plotter.draw_final_dists(xdata,self)
			self.plotter.finished_plotting(self,group)
		else :
			self.plotter = Plotter()
			self.plotter.switch_1=True
			self.plotter.switch_2=True
			self.plotter.draw_only_at_end(self,group,self.plot3_style)

		plt.close('all')

		f_out.write(str(group))
		f_out.write(self.default_delimiter)

		print()
		results_columns=["var", "mean", "median", "stdev","CI68_low","CI68_up","CI95_low","CI95_up"]
		results={}
		for rc in results_columns:
			print(rc+'\t',end='')
			results[rc]=[]
		print()

		nbins=10000
		for i,var in enumerate(self.var_list) :

			xhist, xbins = np.histogram(xdata[i], bins=nbins, range=(0,1), density=True)
			bincentres = [(xbins[i]+xbins[i+1])/2. for i in range(len(xbins)-1)]
			xcum = np.cumsum(xhist)
			xcum /= nbins

			xcum_low = xcum - (0.5-0.6827/2)
			xcum_high = xcum - (0.5+0.6827/2)
			lim_low = np.where(np.diff(np.sign(xcum_low)))[0]
			lim_high = np.where(np.diff(np.sign(xcum_high)))[0]

			err_low = lim_low[0]/nbins
			err_high = lim_high[0]/nbins

			xcum_low = xcum - (0.5-0.9545/2)
			xcum_high = xcum - (0.5+0.9545/2)
			lim_low = np.where(np.diff(np.sign(xcum_low)))[0]
			lim_high = np.where(np.diff(np.sign(xcum_high)))[0]

			err95_low = lim_low[0]/nbins
			err95_high = lim_high[0]/nbins

			results["var"].append(var)
			results["mean"].append(mean(xdata[i]))
			results["median"].append(median(xdata[i]))
			results["stdev"].append(std(xdata[i]))
			results["CI68_low"].append(err_low)
			results["CI68_up"].append(err_high)
			results["CI95_low"].append(err95_low)
			results["CI95_up"].append(err95_high)

			print("f_"+var, "{:.3f}".format(mean(xdata[i])), "{:.3f}".format(median(xdata[i])), "{:.3f}".format(std(xdata[i])), "{:.3f}".format(err_low), "{:.3f}".format(err_high),sep='\t')

			f_out.write("{:.6f}".format(mean(xdata[i])))
			f_out.write(self.default_delimiter)
			f_out.write("{:.6f}".format(median(xdata[i])))
			f_out.write(self.default_delimiter)
			f_out.write("{:.6f}".format(std(xdata[i])))
			f_out.write(self.default_delimiter)
			f_out.write("{:.6f}".format(err_low))
			f_out.write(self.default_delimiter)
			f_out.write("{:.6f}".format(err_high))
			f_out.write(self.default_delimiter)
			f_out.write("{:.6f}".format(err95_low))
			f_out.write(self.default_delimiter)
			f_out.write("{:.6f}".format(err95_high))
			f_out.write(self.default_delimiter)

		# --- Z scores ---
		z_scores, z_pred_scores = self._compute_z_scores()

		def _fmt_z(v):
			return "nan" if np.isnan(v) else "{:.4f}".format(v)

		print()
		print("Z scores — SEM-based (Eqs. 19-20): systematic bias, grows with sqrt(chain length)")
		for iso, z_val in zip(self.isotopes_list, z_scores):
			print("  z_{:s} = {:s}".format(iso, _fmt_z(z_val)))
		print()
		print("Z scores — predictive (chain-length independent): measurement vs. predictive 1-sigma range")
		for iso, z_val in zip(self.isotopes_list, z_pred_scores):
			print("  z_pred_{:s} = {:s}".format(iso, _fmt_z(z_val)))
		print()

		for z_val in z_scores:
			f_out.write("nan" if np.isnan(z_val) else "{:.6f}".format(z_val))
			f_out.write(self.default_delimiter)
		for z_val in z_pred_scores:
			f_out.write("nan" if np.isnan(z_val) else "{:.6f}".format(z_val))
			f_out.write(self.default_delimiter)
		f_out.write("\n")

		print()

		print("LaTex table:")
		for rc in results_columns:
			print(rc+'\t',end=' & ')
		print('\\\\')
		for i,var in enumerate(results["var"]):
			for k in results.keys():
				if k=="var":
					print(results[k][i],end=' & ')
				else:
					print("{:.3f}".format(results[k][i]),end=' & ')
			print('\\\\')

		print()


		for row in self.xb1:
			row.clear()
		for row in self.xd1:
			row.clear()
		for row in self.xb2:
			row.clear()
		for row in self.xd2:
			row.clear()


	def _compute_z_scores(self):
		"""Compute two complementary Z scores per isotope.

		Both scores measure how many standard deviations the measured value lies
		away from the nearest model-consistent solution, but they differ in what
		they use as the "standard deviation":

		z (Eqs. 19-20, PLOS ONE doi:10.1371/journal.pone.0277204)
		  Uses SEM = sigma/sqrt(n) — the precision of the estimated mean.
		  Tests for systematic bias between the model mean and the measurement.
		  Grows with sqrt(n), so longer chains increase apparent significance.

		  * z_i = |x_i - mu_i| / SEM_i                          (no spread, Eq. 19)
		  * z_i = 0                    if |x_i - mu_i| <= Δ_i   (with spread, Eq. 20)
		  * z_i = (|x_i - mu_i| - Δ_i) / SEM_i  otherwise

		z_pred (posterior predictive check, chain-length independent)
		  Uses sigma — the full spread of the model's predictive distribution.
		  Tests whether the measurement falls outside the model's credible range.
		  Invariant to chain length; directly answers "is x outside 1-sigma?".

		  * z_pred_i = |x_i - mu_i| / sigma_i                   (no spread)
		  * z_pred_i = 0                  if |x_i - mu_i| <= Δ_i  (with spread)
		  * z_pred_i = (|x_i - mu_i| - Δ_i) / sigma_i  otherwise

		Common quantities:
		  x_i     = mean measured isotope value for the group
		  mu_i    = mean of the Markov-chain model predictions (xd2[i])
		  sigma_i = std of the Markov-chain model predictions
		  n       = number of chain entries
		  SEM_i   = sigma_i / sqrt(n)
		  Δ_i     = sum_par( |dM/dpar| * spread_par_i ) at chain-mean (f, r)

		Returns: (z_scores, z_pred_scores) — one value per isotope in each list.
		"""
		nan_result = [float('nan')] * self.nisotopes
		if len(self.xd2[0]) < 2:
			return nan_result, nan_result

		# Evaluate derivatives at chain-mean variables
		f = np.array([np.mean(self.xd1[j]) for j in range(self.nsources)])
		r = [np.mean(self.xd1[self.nsources + j]) for j in range(self.nauxvars)]
		aux = self.aux
		S   = self.sources

		z_scores      = []
		z_pred_scores = []
		for i, iso in enumerate(self.isotopes_list):
			x_i     = np.mean([b_k[i] for b_k in self.b])
			mu_i    = np.mean(self.xd2[i])
			sigma_i = np.std(self.xd2[i])
			n_chain = len(self.xd2[i])
			sem_i   = sigma_i / np.sqrt(n_chain) if n_chain > 0 else 0.0

			# Effective half-width from source spreads (Eq. 20)
			delta_i = 0.0
			for par in self.mapPar:
				dM_val   = abs(eval(self.dMdPar[par]))
				delta_i += dM_val * self.spreadPar[par][i]

			diff = abs(x_i - mu_i)

			# --- SEM-based Z (Eqs. 19-20) ---
			if sem_i == 0.0:
				z_i = float('nan')
			elif delta_i > 0.0:
				z_i = 0.0 if diff <= delta_i else (diff - delta_i) / sem_i
			else:
				z_i = diff / sem_i

			# --- Predictive Z (chain-length independent) ---
			if sigma_i == 0.0:
				z_pred_i = float('nan')
			elif delta_i > 0.0:
				z_pred_i = 0.0 if diff <= delta_i else (diff - delta_i) / sigma_i
			else:
				z_pred_i = diff / sigma_i

			z_scores.append(z_i)
			z_pred_scores.append(z_pred_i)

		return z_scores, z_pred_scores

	def run_model(self) :

		self.is_finished=False

		# -------------------------------------------------------------------------------------------------

		# -------------------------------------------------------------------------------------------------
		# -------------------------------------------------------------------------------------------------
		# model

		# fbD*dbD + fnD*dnD + ffD*dfD + fNi*dNi + eps*ln(r) = df 

		# don't print too many decimals
		np.set_printoptions(precision=2)

		# for each loaded dataframe and associated field/lab switch and output data_file
		print('Preparing the input data:')
		print(self.df)

		f_out = open(self.output, "w")

		# colnames refer to the columns written into the output file
		# this includes: auxiliary variables, sources fractions, resulting isotope yields
		colnames= self.var_list

		f_out.write("group"+self.default_delimiter)
		for cl in colnames:
			f_out.write("mean_"+cl+self.default_delimiter)
			f_out.write("median_"+cl+self.default_delimiter)
			f_out.write("stdev_"+cl+self.default_delimiter)
			f_out.write("CI68_low"+cl+self.default_delimiter)
			f_out.write("CI68_up"+cl+self.default_delimiter)
			f_out.write("CI95_low"+cl+self.default_delimiter)
			f_out.write("CI95_up"+cl+self.default_delimiter)
		for iso in self.isotopes_list:
			f_out.write("z_"+iso+self.default_delimiter)
		for iso in self.isotopes_list:
			f_out.write("z_pred_"+iso+self.default_delimiter)
		f_out.write("\n")


		if self.plotting_switch :
			self.plotter = Plotter()
			self.plotter.switch_1=True
			self.plotter.switch_2=True

		print()
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")
		print(" --- Simulation started ---")
		print("Number of iterations:",self.niter)
		print("Number of burn-out MC entries:",self.burnout)
		print("Early termination if the Markov chain already has", self.max_chain_entries, "entries")
		print()
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")

		self.abort=False

		self.ngroups = len(self.df['group'].unique())
		tstart = time.time()

		# for each group in the input data
		for gi,group in enumerate(self.df['group'].unique()) :

			if self.abort==True:
				break

			self.group_i=gi
			self.group=group

			# setting up the plots
			if self.plotting_switch :
				self.plotter.initialize_plots(self,group,self.plot3_style)

			# ----------------------------------







			T=0

			self.data_file = self.output_dir+"/group_"+str(group)

			f_row = open(self.data_file+".csv", "w")
			# header:
			for i in range(self.nauxvars):
				f_row.write(self.aux_vars[i])
				f_row.write(self.default_delimiter)
			for i in range(self.nsources):
				f_row.write(self.sources_list[i])
				f_row.write(self.default_delimiter)
			for i in range(self.nisotopes):
				f_row.write(self.isotopes_list[i])
				f_row.write(self.default_delimiter)
			f_row.write('\n')

			b=[]
			stdev2_data=[0]*self.nisotopes # accumulated stdev^2 calculated from data uncertainties for each isotope

			print()
			print("---------------------------------- GROUP",group,"-------------------------------------------")
			# for each row in the dataframe subset
			for index, row in self.df.loc[self.df['group'] == group].iterrows():
				# read row into vector bi
				bi = np.array([row[self.isotopes_list[i]] for i in range(self.nisotopes)],dtype='double')
				# save vector bi into list b
				b.append(bi)
				print("b: ",bi)

				for i in range(self.nisotopes):
					stdev2_data[i] += row["stdev("+self.isotopes_list[i]+")"]**2

			self.chain_counter=0
			self.burnout_chain_len=0

			start = time.time()

			# run random sampling niter times
			for self.ii in range(self.niter):


				if self.abort==True:
					print("Model terminated by the user.")
					break

				# sample self.sources ratios from Dirichlet with the same means
				f = np.random.dirichlet((1.,)*self.nsources)
				# sample r from the configured prior (uniform by default)
				r = [self._sample_aux_prior(self.aux_vars[i]) for i in range(self.nauxvars)]

				# initialize log-likelihood (unnormalized)
				L=1.

				# metropolis condition:
				#  if L(i+1) < L(i)  ->  accept
				#  if L(i+1) > L(i)  ->  accept with probability = L(i+1/L(i)

				alpha = np.random.uniform(0.0,1.0)

				checkpoint = 100
				if self.ii%checkpoint==0:
					print("iteration:\t",self.ii, "\t chain length:",self.chain_counter, end='\t')

					end = time.time()
					print("time/"+str(checkpoint)+" iters: "+str("{:.3f}".format(end - start))+'s')
					start = time.time()

					if (self.chain_counter >= self.max_chain_entries):
						print("Terminating loop - already have enough in the chains.")
						break



				# just so these eval() instructions can be simpler
				aux=self.aux
				aux_stdev=self.aux_stdev

				# calculate expected measured values (means) using random samples from above

				#model_definition=["M0[i] + E[i] * np.log(r)"],
			        #model_derivatives=["f[0]","f[1]","f[2]","f[3]","np.log(r)"],

				# Calculation of M0[i] = sum_j( S[i]_j*f_j )
				M0 = np.zeros(shape=(self.nisotopes),dtype='double')
				for i in range(self.nisotopes):
					for j in range(self.nsources):
						M0[i] += f[j]*self.sources[i][j]

				S = self.sources

				# Calculation of M
				# using definition given in self.model_definition
				# (blind trust)
				M=np.zeros(shape=(self.nisotopes),dtype='double')
				if self.aux_toggle:
					for i in range(self.nisotopes):
						#print(self.model_definition)
						#print("M0[i]",M0[i])
						#print("aux[0][i]",aux[0][i])
						#print("aux[1][i]",aux[1][i])
						#print("r[0]",r[0])
						#print((M0[i]-aux[0][i]*r[0])*(1-aux[1][i])+8.6*aux[1][i])
						#print()
						M[i] = eval(self.model_definition_code)
				else :
					M = M0.copy()

				# Combined Gaussian sigma: error propagation from source stdevs plus data measurement uncertainty.
				# Each term: (dM/dPar_i * stdev_i)^2, summed in quadrature.
				M_stdev=np.zeros(shape=(self.nisotopes),dtype='double')

				for i in range(self.nisotopes):
					M_stdev[i] += stdev2_data[i]
					for par in self.mapPar:
						M_stdev[i] += (eval(self.dMdPar_code[par])*self.stdevPar[par][i])**2
					M_stdev[i]=np.sqrt(M_stdev[i])

				for i in range(self.nisotopes):
					# Collect the effective half-width of the uniform uncertainty for each parameter.
					# For a source/aux par with spread d, its contribution to the uncertainty in M is
					# Uniform(-|dM/dPar|*d, +|dM/dPar|*d), a symmetric distribution centered at zero.
					uniform_halfwidths = []
					for par in self.mapPar:
						w = abs(eval(self.dMdPar_code[par])) * self.spreadPar[par][i]
						if w > 0:
							uniform_halfwidths.append(w)

					if not uniform_halfwidths:
						# Pure Gaussian path - fast analytical evaluation
						if M_stdev[i] > 0:
							pdf = norm(loc=M[i], scale=M_stdev[i])
							for b_k in b:
								L *= pdf.pdf(b_k[i])
					else:
						# Numerical convolution path.
						# The likelihood is the convolution of:
						#   - one Gaussian (combined stdev from all sources + data uncertainty)
						#   - one symmetric Uniform per parameter with nonzero spread
						# evaluated at (b_k[i] - M[i]).
						#
						# n_conv_points controls the resolution/speed trade-off.
						# Fewer points = faster but coarser; 200 is a practical default.
						n_conv_points = 200

						total_spread = sum(uniform_halfwidths)
						max_eval_dist = float(np.max([abs(b_k[i] - M[i]) for b_k in b]))
						half_width = total_spread + float(np.maximum(5.0 * M_stdev[i], total_spread * 0.1)) + max_eval_dist

						x_conv = np.linspace(-half_width, half_width, n_conv_points)
						dx = x_conv[1] - x_conv[0]

						# Start with the combined Gaussian, or a delta-function approximation when stdev = 0
						if M_stdev[i] > 0:
							compound_pdf = norm(loc=0.0, scale=M_stdev[i]).pdf(x_conv)
						else:
							compound_pdf = np.zeros(n_conv_points)
							compound_pdf[n_conv_points // 2] = 1.0 / dx

					# Sequentially convolve in each uniform component.
					# Using direct numpy array ops avoids scipy object overhead.
					for w in uniform_halfwidths:
						u_pdf = np.where((x_conv >= -w) & (x_conv <= w), 1.0 / (2.0 * w), 0.0)
						compound_pdf = fftconvolve(compound_pdf, u_pdf, mode='same') * dx

						# Renormalize to correct for fftconvolve edge truncation
						norm_factor = np.sum(compound_pdf) * dx
						if norm_factor > 0:
							compound_pdf /= norm_factor

						for b_k in b:
							L *= float(np.interp(b_k[i] - M[i], x_conv, compound_pdf, left=0.0, right=0.0))


				# if Metropolis condition fulfilled

				# additive logL
				#if (L-T) >= np.log(alpha) : 

				# multiplicative L
				#if L/T >= alpha : 
				# changed to this because I wanted T to be zero initially:
				if L >= alpha*T : 
					
					if self.burnout_chain_len>=self.burnout:
						if self.chain_counter%self.chain_checkpoint_every==0:
							tend = time.time()
							print(" --> time/"+str(self.chain_checkpoint_every)+" chain entries: "+str("{:.3f}".format(tend - tstart))+'s')
							tstart = time.time()

							if self.plotting_switch:
								timer_checkpoint_start = time.time()
								self.plotter.draw_at_checkpoint(self)
								timer_checkpoint_end = time.time()
								if(self.verbosity==2) :
									print("checkpoint plotting [s]:",timer_checkpoint_end-timer_checkpoint_start)

						for i in range(self.nauxvars):
							f_row.write( str(r[i])+self.default_delimiter)
						for i in range(self.nsources):
							f_row.write( str(f[i])+self.default_delimiter)
						for i in range(self.nisotopes):
							f_row.write( str(M[i])+self.default_delimiter)
						f_row.write( "\n")
						self.chain_counter+=1

					# update the threshold
					T=L

					# arrays used for plotting:
					for i in range(self.nsources):
						self.xb1[i].append(f[i])
					for i in range(self.nauxvars):
						self.xb1[i+self.nsources].append(r[i])
					for i in range(self.nisotopes):
						self.xb2[i].append(M[i])

					if self.burnout_chain_len>=self.burnout:
						# arrays used for plotting:
						for i in range(self.nsources):
							self.xd1[i].append(f[i])
						for i in range(self.nauxvars):
							self.xd1[i+self.nsources].append(r[i])
						for i in range(self.nisotopes):
							self.xd2[i].append(M[i])
					else:
						self.burnout_chain_len+=1
						if self.burnout_chain_len==self.burnout:
							print("end of burnout")
							tstart = time.time()


					if self.plotting_switch:
						timer_metropolis_start = time.time()
						self.plotter.draw_at_metropolis(self,M,r,f)
						timer_metropolis_end = time.time()

						if(self.verbosity==2) :
							print("metropolis plotting [s]:",timer_metropolis_end-timer_metropolis_start)


			f_row.close()

			# Store current group's measurements so sim_finished can access them for Z scores
			self.b = b

			if (self.chain_counter < self.max_chain_entries and not self.abort):
				print("Terminating - reached limit of max iterations.")
			self.sim_finished(group,f_out)

			print("Finished for group ", group)
			print()

		f_out.close()

		rp = ResultsPlot.ResultsPlot(self.output,self.myfmt)

		self.is_finished=True
		if not self.abort:
			print("All finished successfully!")
			print()
















