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


	def initialize_plots(self,model,group):
		# ion for live updates
		if self.switch_1 and self.switch_2:
			plt.ion()
		else:
			return

		if self.switch_1:
			# figure 1 -- the chains
			self.plot1 = TimeSeriesPlot.TimeSeriesPlot(model)
			# figure 2 -- the path on the isotope map
			self.plot2 = PathPlot.PathPlot(model,group)

		if self.switch_2 :
			# figure 3 -- histograms and correaltion of Markov chain variables`
			self.plot3 = CorrelationPlot.CorrelationPlot(model)

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
			self.plot1.save(model.output_dir+'/'+"group_"+str(group)+"_")
			self.plot2.save(model.output_dir+'/'+"group_"+str(group)+"_")
		if self.switch_2:
			self.plot3.save(model.output_dir+'/'+"group_"+str(group)+"_")

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

	def draw_only_at_end(self,model,group):
		self.switch_1=False
		self.initialize_plots(model,group)
		self.draw_at_checkpoint(model)
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

		self.abort = False
		self.latex = False

		self.chain_counter=0

		self.axes_labels=[]
		self.default_delimiter=','

		# was the 'set_up_data' method called?
		self.initialized=False

		# 0 - just the regular output
		# 1 - technical and mathematical details
		# 2 - coding debug
		self.verbosity=1

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
			if var!="group" and var!="label" and "sigma" not in var and var!="sample_id":
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

		delta_sum=0
		for iso in self.isotopes_list:
			for delta in self.df_sources["delta("+iso+")"]:
				delta_sum += delta
		if delta_sum == 0:
			print("Using gaussian-like likelihood function.")
			self.erf_toggle = False
		else :
			print("Using erf-like likelihood function.")
			


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
		self.output_dir=self.output_dir+self.dataset_name
		Path(self.output_dir).mkdir(parents=True, exist_ok=True)
		self.output=self.output_dir+'/results.csv'
		print("Output file: ")
		print(self.output)
		print()

	def set_iterations(self, niter=1e7, burnout=1e5, max_chain_entries=500) :

		# number of iterations, burnout, desired Markov chain length
		self.niter=int(niter)
		self.burnout=int(burnout)
		self.max_chain_entries=int(max_chain_entries)

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
			self.model_definition = re.sub('M0\[i\]',strM0,self.model_definition)

			self.sym_model_definition = re.sub(r'\[(\d*)\]',r'\1',self.model_definition)
			self.sym_model_definition = re.sub('\[i\]','',self.sym_model_definition)

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
		self.sources_delta=np.zeros(shape=(self.nisotopes,self.nsources),dtype='double')
		for i,iso in enumerate(self.isotopes_list):
			for j,src in enumerate(self.sources_list):
				self.sources[i][j]=self.df_sources.loc[ (self.df_sources['source']==src) ].iloc[0][iso]
				self.sources_stdev[i][j]=self.df_sources.loc[ (self.df_sources['source']==src) ].iloc[0]["sigma("+iso+")"]
				self.sources_delta[i][j]=self.df_sources.loc[ (self.df_sources['source']==src) ].iloc[0]["delta("+iso+")"]

		print("self.sources")
		print(self.sources)
		print("self.sources_stdev")
		print(self.sources_stdev)
		print("self.sources_delta")
		print(self.sources_delta)
		print()
				
		if self.aux_toggle:
			print('Numer of auxiliary parameters:',len(self.aux_pars))
			print(self.aux_pars)

			print('Numer of auxiliary variables:',self.nauxvars)

			self.aux=np.zeros(shape=(self.nisotopes,self.nauxpars),dtype='double')
			self.aux_stdev=np.zeros(shape=(self.nisotopes,self.nauxpars),dtype='double')
			self.aux_delta=np.zeros(shape=(self.nisotopes,self.nauxpars),dtype='double')
			for i,iso in enumerate(self.isotopes_list):
				for j,nm in enumerate(self.aux_pars):
					self.aux[i][j]=self.df_aux.loc[ (self.df_aux['name']==nm)].iloc[0][iso]
					self.aux_stdev[i][j]=self.df_aux.loc[ (self.df_aux['name']==nm) ].iloc[0]["sigma("+iso+")"]
					self.aux_delta[i][j]=self.df_aux.loc[ (self.df_aux['name']==nm) ].iloc[0]["delta("+iso+")"]

			print("self.aux:", self.aux)
			print("self.aux_stdev:", self.aux_stdev)
			print("self.aux_delta:", self.aux_delta)
			print()

		if self.model_derivatives==[]:
			for i in range(self.nsources):
				self.model_derivatives.append("f["+str(i)+"]")

		self.mapPar={}
		self.sigmaPar={}
		self.dMdPar={}
		print("----")
		print(self.sources_list)
		print(self.model_derivatives)
		for j in range(self.nsources) :
			self.mapPar[self.sources_list[j]] = [self.sources[i][j] for i in range(self.nisotopes)]
			self.sigmaPar[self.sources_list[j]] = [self.sources_stdev[i][j] for i in range(self.nisotopes)]
			self.dMdPar[self.sources_list[j]] = self.model_derivatives[j]
		for j in range(self.nauxpars) :
			self.mapPar[self.aux_pars[j]] = [self.aux[i][j] for i in range(self.nisotopes)]
			self.sigmaPar[self.aux_pars[j]] = [self.aux_stdev[i][j] for i in range(self.nisotopes)]
			self.dMdPar[self.aux_pars[j]] = self.model_derivatives[self.nsources + j]

		print("self.mapPar: ",self.mapPar)
		print("self.sigmaPar: ",self.sigmaPar)
		print("self.dMdPar: ",self.dMdPar)
		print()

		self.xd1 = [ [] for i in range(self.nvariables)]

		if len(self.axes_labels)==0:
			self.axes_labels = self.isotopes_list.copy()


	def sim_finished(self,group,f_out) :
		xdata=self.xd1
		print("Simulation has finished.")

		print("Output directory:", self.data_file)

		print("Results:")
		print("var.", "mean", "median", "stdev","lim_low","lim_up",sep='\t')

		if self.plotting_switch :
			self.plotter.draw_final_dists(xdata,self)
			self.plotter.finished_plotting(self,group)
		else :
			self.plotter = Plotter()
			self.plotter.draw_only_at_end(self,group)

		plt.close('all')

		nbins=10000
		f_out.write(str(group)+self.default_delimiter)
		for i,var in enumerate(self.var_list) :

			xhist, xbins = np.histogram(xdata[i], bins=nbins, range=(0,1), density=True)
			bincentres = [(xbins[i]+xbins[i+1])/2. for i in range(len(xbins)-1)]
			xcum = np.cumsum(xhist)
			xcum /= nbins

			xcum_low = xcum - (0.5-0.341)
			xcum_high = xcum - (0.5+0.341)
			lim_low = np.where(np.diff(np.sign(xcum_low)))[0]
			lim_high = np.where(np.diff(np.sign(xcum_high)))[0]

			err_low = lim_low[0]/nbins
			err_high = lim_high[0]/nbins

			print("f_"+var, "{:.4f}".format(mean(xdata[i])), "{:.4f}".format(median(xdata[i])), "{:.4f}".format(std(xdata[i])), "{:.4f}".format(err_low), "{:.4f}".format(err_high),sep='\t')

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

		print()
		f_out.write("\n")

		for row in self.xd1:
			row.clear()


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
		#print("var.", "mean", "median", "stdev","lim_low","lim_up",sep='\t')
		for cl in colnames:
			f_out.write("mean_"+cl+self.default_delimiter)
			f_out.write("median_"+cl+self.default_delimiter)
			f_out.write("stdev_"+cl+self.default_delimiter)
			f_out.write("lim_low_"+cl+self.default_delimiter)
			f_out.write("lim_up_"+cl+self.default_delimiter)
		f_out.write("\n")


		if self.plotting_switch :
			self.plotter = Plotter()
			self.plotter.switch_1=self.plotting_switch
			self.plotter.switch_2=self.plotting_switch

		print()
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")
		print(" --- Simulation started ---")
		print("Number of iterations:",self.niter)
		print("Number of burn-out iterations:",self.burnout)
		print("Early termination if the Markov chain already has", self.max_chain_entries, "entries")
		print()
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")
		print("--------------------------------------------------------------------------------------------------------")

		self.abort=False

		self.ngroups = len(self.df['group'].unique())

		# for each group in the input data
		for gi,group in enumerate(self.df['group'].unique()) :

			if self.abort==True:
				break

			self.group_i=gi
			self.group=group

			for row in self.xd1:
				row.clear()

			# setting up the plots
			if self.plotting_switch :
				self.plotter.initialize_plots(self,group)

			# ----------------------------------







			T=0

			maxL=-1

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
			sigma2_data=[0]*self.nisotopes # accumulated sigma^2 calculated from data uncertainties for each isotope

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
					sigma2_data[i] += row["sigma("+self.isotopes_list[i]+")"]**2

			self.chain_counter=0

			start = time.time()


			# run random sampling niter times
			for self.ii in range(self.niter):


				if self.abort==True:
					print("Model terminated by the user.")
					break

				# sample self.sources ratios from Dirichlet with the same means
				f = np.random.dirichlet((1.,)*self.nsources)
				# sample r from uniform distribution
				r = [np.random.uniform() for i in range(0,self.nauxvars)]

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

					if(self.ii==self.burnout):
						print("end of burnout")
						tstart = time.time()

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
						M[i] = eval(self.model_definition)
				else :
					M = M0.copy()

				# calculate standard deviations
				# 
				# it has a form of: sqrt( sum_i( dM/dPar_i sigma(Par_i) )  )
				# plus contribution from measurement errors
				
				M_stdev=np.zeros(shape=(self.nisotopes),dtype='double')

				for i in range(self.nisotopes):
					# measured data uncertainty
					M_stdev[i] += sigma2_data[i]

					for par in self.mapPar:
						M_stdev[i] += (eval(self.dMdPar[par])*self.sigmaPar[par][i])**2
						
					M_stdev[i]=np.sqrt(M_stdev[i])

				for i in range(self.nisotopes):
					# Erf-like likelihood
					if self.erf_toggle:
						sumdS=0
						for j,src in enumerate(self.sources_list):
							sumdS+=eval(self.dMdPar[src]) * self.sources_delta[i][j]
						for j,par in enumerate(self.aux_pars):
							sumdS+=eval(self.dMdPar[par]) * self.aux_delta[i][j]
						for b_k in b:
							if sumdS==0:
								L=0
							else:
								L *= (1./sumdS) * (  math.erf( (sumdS+M[i]-b_k[i]) / (sqrt(2)*M_stdev[i]) ) - math.erf( (-sumdS+M[i]-b_k[i]) / (sqrt(2)*M_stdev[i]) )  )

					# Gaussian-like likelihood
					else :
						for b_k in b:
							L *= (1./M_stdev[i]/np.sqrt(2.*np.pi)) * np.exp( -(b_k[i]-M[i])**2./(2.*M_stdev[i]) )


				# if Metropolis condition fulfilled

				# additive logL
				#if (L-T) >= np.log(alpha) : 

				# multiplicative L
				#if L/T >= alpha : 
				# changed to this because I wanted T to be zero initially:
				if L >= alpha*T : 
					
					if self.ii>self.burnout:
						if self.chain_counter%10==0:
							tend = time.time()
							print(" --> time/"+"10 chain entries: "+str("{:.3f}".format(tend - tstart))+'s')
							tstart = time.time()

							if self.plotting_switch and self.ii > self.burnout:
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

					if self.ii>self.burnout:
						# arrays used for plotting:
						for i in range(self.nsources):
							self.xd1[i].append(f[i])
						for i in range(self.nauxvars):
							self.xd1[i+self.nsources].append(r[i])

					if self.plotting_switch:
						timer_metropolis_start = time.time()
						self.plotter.draw_at_metropolis(self,M,r,f)
						timer_metropolis_end = time.time()
						if(self.verbosity==2) :
							print("metropolis plotting [s]:",timer_metropolis_end-timer_metropolis_start)


			f_row.close()

			if (self.chain_counter < self.max_chain_entries and not self.abort):
				print("Terminating - reached limit of max iterations.")
			self.sim_finished(group,f_out)

			print("Finished for group ", group)
			print()

		f_out.close()
		self.is_finished=True
		if not self.abort:
			print("All finished successfully!")
			print()
















