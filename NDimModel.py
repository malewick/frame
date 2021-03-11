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
import sys
import re
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rcParams['text.usetex'] = True

import matplotlib.animation as animation

import time

from matplotlib.patches import Rectangle


class Model:

	def __init__(self):

		# online plotting, True by default
		self.plotting_switch_1=True
		self.plotting_switch_2=True
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

	def load_from_fielnames(self, data_file, sources_file, aux_file=None):
		self.load_measurements(data_file)
		self.load_sources(sources_file)
		if aux_file is not None:
			self.load_aux(aux_file)

		self.set_up_data()

	def load_measurements(self, data_file):

		# Create measurements DataFrame
		self.data_file = data_file
		self.df = pd.read_csv(data_file, skiprows = 0, delimiter="[,\t]", engine='python')
		print("Measurements dataframe:")
		print(self.df)

		self.isotopes_list=[]
		for var in [*self.df.columns]:
			if var!="group" and var!="label" and "sigma" not in var:
				self.isotopes_list.append(var)
		print("isotopes_list: ",self.isotopes_list)
		print()

	def load_sources(self, sources_file):

		# Create sources DataFrame 
		self.df_sources = pd.read_csv(sources_file, skiprows = 0, delimiter="[,\t]", engine='python')
		print("Sources dataframe:")
		print(self.df_sources)

		self.sources_list=[]
		for var in self.df_sources["source"].unique():
			self.sources_list.append(var)
		print("sources list:", self.sources_list)
		print()


	def load_aux(self, aux_file):

		# Create auxiliary parameter DataFrame
		print("--- Aux file import ---")

		with open(aux_file) as f:
			content = f.readlines()
		content = [x.strip() for x in content]

		aux_par_names = content[0].split()
		print("aux_par_names:",aux_par_names)

		aux_var_names = content[1].split()
		print("aux_var_names:",aux_var_names)

		model_equation = content[2]
		print("model_equation: ",model_equation)

		if "," in content[3]:
			derivatives = content[3].split(",")
		else:
			derivatives = content[3].split("\t")
		print("derivatives:", derivatives)

		self.df_aux = pd.read_csv(aux_file, skiprows = 4, delimiter='[,\t]', engine='python')
		print("Auxiliary parameter DataFrame")
		print(self.df_aux)
		print()

		self.model_definition=model_equation
		self.model_derivatives=derivatives
		self.aux_pars=aux_par_names
		self.aux_vars=aux_var_names

		self.aux_toggle=True



	def set_outdir(self, out_dir=".") :

		# Output directory and files
		self.output_dir=out_dir
		self.dataset_name=re.sub('.*/', '', re.sub('.csv', '', self.data_file))
		self.output=self.output_dir+self.dataset_name+'.csv'
		print("Output file: ")
		print(self.output)
		print()

	def set_iterations(self, niter=1e7, burnout=1e5, max_chain_entries=500) :

		# number of iterations, burnout, desired Markov chain length
		self.niter=int(niter)
		self.burnout=int(burnout)
		self.max_chain_entries=int(max_chain_entries)

	def set_up_data(self) :

		if self.aux_toggle :
			model_definition_preregex = self.model_definition
			model_derivatives_preregex = self.model_derivatives.copy()
			model_aux_pars_set = set(re.findall('[A-Z]',model_definition_preregex))
			for char in ['S','M']:
				if char in model_aux_pars_set: model_aux_pars_set.remove(char)
			if model_aux_pars_set != set(self.aux_pars) :
				print("Scream WTF")

			for i, auxpar in enumerate(self.aux_pars):
				self.model_definition = re.sub(r'\b'+str(auxpar)+r'\b','aux['+str(i)+']',self.model_definition)
				for j in range(len(self.model_derivatives)) :
					self.model_derivatives[j] = re.sub(r'\b'+str(auxpar)+r'\b','aux['+str(i)+']',self.model_derivatives[j])
			for i, auxvar in enumerate(self.aux_vars):
				self.model_definition = re.sub(r'\b'+str(auxvar)+r'\b','r['+str(i)+']',self.model_definition)
				for j in range(len(self.model_derivatives)) :
					self.model_derivatives[j] = re.sub(r'\b'+str(auxvar)+r'\b','r['+str(i)+']',self.model_derivatives[j])

			print()
			print("Model equation")
			print("Before regex:\t",model_definition_preregex)
			print("After regex:\t",self.model_definition)
			print()
			print("and derivatives")
			print("Before regex:\t",model_derivatives_preregex)
			print("After regex:\t",self.model_derivatives)
			print()

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
		for i,iso in enumerate(self.isotopes_list):
			for j,src in enumerate(self.sources_list):
				self.sources[i][j]=self.df_sources.loc[ (self.df_sources['isotope']==iso) & (self.df_sources['source']==src) ].iloc[0]['value']
				self.sources_stdev[i][j]=self.df_sources.loc[ (self.df_sources['isotope']==iso) & (self.df_sources['source']==src) ].iloc[0]['stdev']

		print("self.sources")
		print(self.sources)
		print("self.sources_stdev")
		print(self.sources_stdev)
		print()
				
		if self.aux_toggle:
			print('Numer of auxiliary parameters:',len(self.aux_pars))
			print(self.aux_pars)

			print('Numer of auxiliary variables:',self.nauxvars)

			self.aux=np.zeros(shape=(self.nauxpars,self.nisotopes),dtype='double')
			self.aux_stdev=np.zeros(shape=(self.nauxpars,self.nisotopes),dtype='double')
			for j,nm in enumerate(self.aux_pars):
				for i,iso in enumerate(self.isotopes_list):
					self.aux[j][i]=self.df_aux.loc[ (self.df_aux['isotope']==iso) & (self.df_aux['name']==nm)].iloc[0]['value']
					self.aux_stdev[j][i]=self.df_aux.loc[ (self.df_aux['isotope']==iso) & (self.df_aux['name']==nm) ].iloc[0]['stdev']

			print("self.aux:", self.aux)
			print("self.aux_stdev:", self.aux_stdev)
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
			self.mapPar[self.aux_pars[j]] = [self.aux[j][i] for i in range(self.nisotopes)]
			self.sigmaPar[self.aux_pars[j]] = [self.aux_stdev[j][i] for i in range(self.nisotopes)]
			self.dMdPar[self.aux_pars[j]] = self.model_derivatives[self.nsources + j]

		print("self.mapPar: ",self.mapPar)
		print("self.sigmaPar: ",self.sigmaPar)
		print("self.dMdPar: ",self.dMdPar)
		print()


	def run_model(self) :

		# -------------------------------------------------------------------------------------------------
		# setting up the plots

		# ion for live updates
		if self.plotting_switch_1 or self.plotting_switch_2:
			plt.ion()

		if self.plotting_switch_1:
			# figure 1 -- the chains
			fig, ax = plt.subplots(self.nvariables, sharex=True)

		if self.plotting_switch_2:
			# figure 2 -- the 1D histograms and correlation plots
			fig1d, ax1d = plt.subplots(self.nvariables,self.nvariables,figsize=(2*self.nvariables,2*self.nvariables))
			fig1d.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.90, wspace=0., hspace=0.)

		if self.plotting_switch_1:
			# figure 3 -- 2D path of isotope signatures
			fig2d, ax2d = plt.subplots(1,2*self.nisotopes-3,figsize=(1+(2*self.nisotopes-3)*4,4))
			fig2d.subplots_adjust(left=0.15/(2*self.nisotopes-3),right=0.95,bottom=0.13,top=0.95)
			if not hasattr(ax2d, "__getitem__") :
				ax2d = [ax2d]

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
		colnames= self.aux_vars
		colnames.extend(self.sources_list)
		colnames.extend(self.isotopes_list)

		f_out.write("group\t")
		f_out.write("case\t")
		for cl in colnames:
			f_out.write("mean_"+cl+"\t")
			f_out.write("stdev_"+cl+"\t")
		f_out.write("\n")

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

		# for each group in the input data
		for group in self.df['group'].unique() :

			# let's first set up the plots

			# beware of this:
			# 
			# [stackoverflow.com/questions/2397141]
			# 
			# >>> a = [[0]*3]*3
			# >>> a
			# [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
			# >>> a[0][0]=1
			# >>> a
			# [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
			# 
			# 
			# this works fine though: t = [ [0]*3 for i in range(3)]
			# ----------------------------------

			if self.plotting_switch_1 or self.plotting_switch_2:

				xdata = [0]
				ydata = [ [0] for i in range(self.nvariables)]

				xd2 = [ [] for i in range(self.nisotopes)]

				xmin=[]
				xmax=[]
				for i in range(self.nisotopes):
					xmin.append(min(self.sources[i])-7)
					xmax.append(max(self.sources[i])+7)

			if self.plotting_switch_1 :

				line=[]
				for i, a in enumerate(ax):
					line_temp, = a.plot(xdata, ydata[i], linestyle='-', linewidth=1, alpha=1)
					#a.set_xlim(0,1)
					a.set_ylim(0,1)
					a.set_ylabel(self.var_list[i])
					line.append(line_temp)



				# ----------------------------------


				ld2 = []

				#isotopes_list=["SP","d18O","d15N"]
				counter=0
				for i in range(self.nisotopes):
					for j in range(i+1,self.nisotopes):
						ax2d[counter].set_xlim(xmin[i],xmax[i])
						ax2d[counter].set_ylim(xmin[j],xmax[j])
						ax2d[counter].set_xlabel(self.isotopes_list[i]+r" [permil]")
						ax2d[counter].set_ylabel(self.isotopes_list[j]+r" [permil]")

						dfs=self.df.loc[self.df['group']==group]
						ax2d[counter].errorbar(dfs[self.isotopes_list[i]].tolist(), dfs[self.isotopes_list[j]].tolist(),
								       xerr=dfs["sigma("+self.isotopes_list[i]+")"].tolist(),
								       yerr=dfs["sigma("+self.isotopes_list[j]+")"].tolist(),
								       ecolor="black",
								       mec="None",
								       mfc="black",
								       marker="o",
								       linestyle='',
								       alpha=1.,
								)


						for k in range(self.nsources) :
							vx=self.sources[i][k]
							sx=self.sources_stdev[i][k]
							vy=self.sources[j][k]
							sy=self.sources_stdev[j][k]
							rect = Rectangle((vx-sx,vy-sy),2*sx,2*sy,linewidth=1,edgecolor='black',facecolor="black",alpha=0.2)
							ax2d[counter].add_patch(rect)
							ax2d[counter].text(vx-sx+0.5,vy-sy+0.5, self.sources_list[k], fontsize=12)

						line_temp, = ax2d[counter].plot(xd2[i],xd2[j], linestyle='-', linewidth=1, alpha=1)
						ld2.append(line_temp)

						counter+=1

			# ----------------------------------

			if self.plotting_switch_2 :

				xd1 = [ [] for i in range(self.nvariables)]
				ld1 = []

				# 1D histograms
				for i in range(self.nvariables):
					hist_temp = ax1d[i,i].hist(xd1[i], bins=25, range=(0,1))
					ld1.append(hist_temp)

				# 2D correlations plots
				for i in range(self.nvariables-1):
					for j in range(i+1,self.nvariables):
						ax1d[i,j].plot(xd1[i],xd1[j],marker='o',linestyle='none',markersize=0.4)

				# ----------------------------------


				xd3 = [ [] for i in range(self.nisotopes)]
				ld3 = []

			# ----------------------------------









			T=-1.

			maxL=-1

			Path(self.output_dir+"grouped_chains/"+self.dataset_name+"/").mkdir(parents=True, exist_ok=True)
			self.data_file = self.output_dir+"grouped_chains/"+self.dataset_name+"/"+str(group)

			f_row = open(self.data_file+".dat", "w")

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

			chain_counter=0

			start = time.time()


			# run random sampling niter times
			for ii in range(self.niter):

				# sample self.sources ratios from Dirichlet with the same means
				f = np.random.dirichlet((1.,)*self.nsources)
				# sample r from uniform distribution
				r = [np.random.uniform() for i in range(0,self.nauxvars)]

				# initialize log-likelihood (unnormalized)
				L=1.

				# metropolis condition:
				#  if L(i+1) < L(i)  ->  accept
				#  if L(i+1) > L(i)  ->  accept with probability = L(i+1/L(i)

				# for faster burnout the alpha is set to 1 in this period
				alpha = 1 

				if (ii>self.burnout):
					alpha = np.random.uniform(0.0,1.0)

				if ii%10000==0:
					print("iteration:\t",ii, "\t chain length:",chain_counter)

					end = time.time()
					print("time/10k iters:",end - start)
					print("time/iteration:",(end - start)/10000)
					start = time.time()

					if(ii==self.burnout):
						print("end of burnout")

					#############################################################################
					# plotting 

					if self.plotting_switch_2 and ii > self.burnout and len(xd1[0])>1:

						# 1D histograms
						for i in range(self.nvariables):
							ax1d[i,i].clear()
						for i in range(self.nvariables):
							ld1[i] = ax1d[i,i].hist(xd1[i], bins=25, color='C0', range=(0,1))
							ax1d[i,i].set_xlim(-0.1,1.15)
							#ax1d[i,i].text(0.7, 0.7,self.var_list[i],color='black',size=22,usetex=True,family='serif',transform=ax1d[i,i].transAxes)
							ax1d[i,i].text(0.7, 0.7,self.var_list[i],color='black',size=22,family='serif',transform=ax1d[i,i].transAxes)
						# 2D correlations plots
						for i in range(self.nvariables-1):
							for j in range(i+1,self.nvariables):
								ax1d[i,j].clear()
								#ax1d[i,j].hexbin(xd1[i], xd1[j], gridsize=30, bins='log', cmap='viridis', extent=[0,1,0,1])
								ax1d[i,j].plot(xd1[j],xd1[i],marker='o',linestyle='none',color='black', markersize=0.4, alpha=0.5)
								ax1d[i,j].set_xlim(-0.1,1.15)
								ax1d[i,j].set_ylim(-0.1,1.15)

						# 2D correlations text
						for i in range(1,self.nvariables):
							for j in range(0,i):
								ax1d[i,j].clear()
								corr = np.corrcoef(xd1[i],xd1[j])[0,1]
								clr=""
								if corr<0 :
									clr="C0"
								elif corr>0:
									clr="C3"
								sz=abs(corr)*10.+7.
								#ax1d[i,j].text(0.3, 0.3,str("{:.2f}".format(corr)),color=clr,size=sz,usetex=True,family='serif')
								ax1d[i,j].text(0.3, 0.3,str("{:.2f}".format(corr)),color=clr,size=sz,family='serif')
								ax1d[i,j].set_xlim(-0.1,1.15)

						# setting up ticks and labels
						for i in range(self.nvariables-1):
							ax1d[i,self.nvariables-1].yaxis.tick_right()

						for j in range(0,self.nvariables):
							ax1d[0,j].xaxis.tick_top()

						for i in range(self.nvariables-1):
							for j in range(i+1,self.nvariables-1):
								#ax1d[i,j].set_xticklabels([])
								#ax1d[i,j].set_xticks([])
								ax1d[i,j].set_yticklabels([])
								ax1d[i,j].set_yticks([])

						# 2D correlations text
						for i in range(1,self.nvariables-1):
							for j in range(0,i):
								ax1d[i,j].set_xticklabels([])
								ax1d[i,j].set_xticks([])
						for i in range(1,self.nvariables):
							for j in range(0,i):
								ax1d[i,j].set_yticklabels([])
								ax1d[i,j].set_yticks([])

						for i in range(1,self.nvariables-1):
							ax1d[i,i].set_xticklabels([])
							ax1d[i,i].set_xticks([])
						for i in range(self.nvariables):
							yticks = ax1d[i,i].yaxis.get_major_ticks() 
							yticks[0].label1.set_visible(False)

						fig1d.canvas.draw()
						fig1d.canvas.flush_events()

					#############################################################################

					if (chain_counter >= self.max_chain_entries):
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
				#
				M_stdev=np.zeros(shape=(self.nisotopes),dtype='double')

				for i in range(self.nisotopes):
					M_stdev[i] += sigma2_data[i] # recent addition! (23.02.2021)

					for par in self.mapPar:
						M_stdev[i] += (eval(self.dMdPar[par])*self.sigmaPar[par][i])**2
						
					M_stdev[i]=np.sqrt(M_stdev[i])

				# for each row in the dataframe subset
				for b_k in b:
					for i in range(self.nisotopes):
						# multiplicative L
						#L*=(1./M_stdev[i]/np.sqrt(2.*np.pi)) * np.exp( -(b_k[i]-M[i])**2./(2.*M_stdev[i]) )
						# additive logL
						L += -np.log(M_stdev[i]) - (b_k[i]-M[i])**2./(2.*M_stdev[i])

				# initialize the threshold
				if(ii==0):
					T=L


				# if Metropolis condition fulfilled
				if (L-T) >= np.log(alpha) : 


					#print("M0",M0)
					#print("M",M)
					#print("b",b)
					#print("f",f)
					#print("r",r)
					#print()


					if ii>self.burnout:
						for i in range(self.nauxvars):
							f_row.write( str(r[i])+"\t")
						for i in range(self.nsources):
							f_row.write( str(f[i])+"\t")
						for i in range(self.nisotopes-1):
							f_row.write( str(M[i])+"\t")
						f_row.write( str(M[-1]))
						f_row.write( "\n")
						chain_counter+=1

					# update the threshold
					T=L

					if self.plotting_switch_1 or self.plotting_switch_2:

						# arrays used for plotting:
						xdata.append(xdata[-1]+1)
						for i in range(self.nauxvars):
							ydata[i].append(r[i])
						for i in range(self.nsources):
							ydata[i+self.nauxvars].append(f[i])
						for i in range(self.nisotopes):
							xd2[i].append(M[i])

						if ii>self.burnout:
							# arrays used for plotting:
							for i in range(self.nauxvars):
								xd1[i].append(r[i])
							for i in range(self.nsources):
								xd1[i+self.nauxvars].append(f[i])

						if self.plotting_switch_1 :

							# these are quick to update
							for i, yi in enumerate(ydata):
								line[i].set_xdata(xdata)
								line[i].set_ydata(yi)
								ax[i].set_xlim(0,xdata[-1])
								ax[i].set_ylim(0,1.05)
							fig.canvas.draw()
							fig.canvas.flush_events()

							if len(xd2[0]) > 1:
								counter=0
								for i in range(self.nisotopes):
									for j in range(i+1,self.nisotopes):
										ax2d[counter].plot([xd2[i][-2],xd2[i][-1]],[xd2[j][-2],xd2[j][-1]],color='C0',alpha=0.05)
										ax2d[counter].plot([M[i]],[M[j]],color='C0',alpha=0.5,marker='o',markersize=0.2, linestyle='')
										counter+=1
							fig2d.canvas.draw()
							fig2d.canvas.flush_events()

			f_row.close()


			# calculate the means and stddevs from the markov chains that we've saved before
			df_temp = pd.read_csv(self.data_file+".dat", skiprows = 0, names=colnames, header=None, delimiter="\t")
			means={}
			stdevs={}
			for cl in colnames:
				means[cl]=df_temp[cl].mean()
				stdevs[cl]=df_temp[cl].std()


			# in the final output file save firstly the interation which yielded the highest likelihood
			# and then also the means and stddevs obtained from Markov chain
			f_out.write(str(group)+"\t")
			for cl in colnames:
				f_out.write(str(means[cl])+"\t")
				f_out.write(str(stdevs[cl])+"\t")
			f_out.write("\n")

			if self.plotting_switch_1:
				fig.savefig(self.data_file+"_convergence.png")
				fig2d.savefig(self.data_file+"_path2D.png")
				for i in range(self.nvariables):
					ax[i].clear()
				for i in range(len(ax2d)):
					ax2d[i].clear()
			if self.plotting_switch_2:
				fig1d.savefig(self.data_file+"_hist1D.png")
				for i in range(self.nvariables):
					for j in range(self.nvariables):
						ax1d[i,j].clear()

		f_out.close()
















