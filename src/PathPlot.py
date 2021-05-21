import pandas as pd
import numpy as np
from numpy import *
import re

import matplotlib.pyplot as plt
import matplotlib

from matplotlib.patches import Rectangle


class PathPlot :
	def __init__(self,model,group):
		self.base_size=1.5
		self.xd2 = [ [] for i in range(model.nisotopes)]

		xmin=[]
		xmax=[]
		for i in range(model.nisotopes):
			xmin.append(min(model.sources[i])-max(max(model.sources_spread[i]),max(model.sources_stdev[i]))-2)
			xmax.append(max(model.sources[i])+max(max(model.sources_spread[i]),max(model.sources_stdev[i]))+2)

		self.fig, self.ax = plt.subplots(1,2*model.nisotopes-3,figsize=(((2*model.nisotopes-3))*2*self.base_size,2*(self.base_size)))
		self.fig.subplots_adjust(left=0.22/(2*model.nisotopes-3),right=0.99,bottom=0.2,top=0.98, wspace=0.3)
		if not hasattr(self.ax, "__getitem__") :
			self.ax = [self.ax]

		# ----------------------------------
		self.ld2 = []
		self.ld22 = []

		dfs=model.df.loc[model.df['group']==group]
		counter=0
		for i in range(model.nisotopes):
			for j in range(i+1,model.nisotopes):
				#self.ax[counter].set_xlim(xmin[i],xmax[i])
				#self.ax[counter].set_ylim(xmin[j],xmax[j])
				self.ax[counter].set_xlabel(model.axes_labels[i])
				self.ax[counter].set_ylabel(model.axes_labels[j])

				self.ax[counter].errorbar(dfs[model.isotopes_list[i]].tolist(), dfs[model.isotopes_list[j]].tolist(),
						       xerr=dfs["stdev("+model.isotopes_list[i]+")"].tolist(),
						       yerr=dfs["stdev("+model.isotopes_list[j]+")"].tolist(),
						       ecolor="black",
						       mec="None",
						       mfc="black",
						       marker="o",
						       linestyle='',
						       alpha=1.,
						)

				self.ax[counter].errorbar(model.sources[i],model.sources[j],
						       xerr=model.sources_stdev[i],
						       yerr=model.sources_stdev[j],
						       ecolor="black",
						       mec="None",
						       mfc="None",
						       marker="o",
						       linestyle='',
						       alpha=1.,
						)


				for k in range(model.nsources) :
					vx=model.sources[i][k]
					sx=model.sources_spread[i][k]
					vy=model.sources[j][k]
					sy=model.sources_spread[j][k]
					rect = Rectangle((vx-sx,vy-sy),2*sx,2*sy,linewidth=1,edgecolor='black',facecolor="black",alpha=0.2)
					self.ax[counter].add_patch(rect)
					self.ax[counter].text(vx-sx+0.5,vy-sy+0.5, model.sources_list[k], fontsize=12)

				line_temp, = self.ax[counter].plot(self.xd2[i],self.xd2[j], linestyle='-', linewidth=1, alpha=1)
				self.ld2.append(line_temp)
				line_temp2, = self.ax[counter].plot([],[],color='C0',alpha=0.5,marker='o',markersize=0.2, linestyle='')
				self.ld22.append(line_temp2)

				self.ax[counter].autoscale()

				counter+=1

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

		if model.nauxvars==1:
			counter=0
			for i in range(model.nisotopes):
				for j in range(i+1,model.nisotopes):
					self.ax[counter].autoscale(False)
					self.plot_fractionation(model,i,j,self.ax[counter])
					counter+=1

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()


	def plot_fractionation(self, model, arg1, arg2, axes):

		# shows linescorresponding to the fractionation process
		# makes sense only with one auxiliary variable
		if model.nauxvars != 1:
			return

		# in order to overwrite autoscale
		xlimlow = axes.get_xlim()[0]
		xlimhigh = axes.get_xlim()[1]
		ylimlow = axes.get_ylim()[0]
		ylimhigh = axes.get_ylim()[1]

		# taking symbolic equation for the model and regexing the indices
		model_definition = model.sym_model_definition
		model_definition = re.sub(r'f(\d*)',r'f[\1]',model_definition)
		model_definition = re.sub(r'r(\d*)',r'r[\1]',model_definition)
		model_definition = re.sub(r'aux(\d*)',r'aux[i][\1]',model_definition)
		model_definition = re.sub(r'S(\d*)',r'S[i][\1]',model_definition)

		# first initialize with default f to get the a=dy/dx 
		f=[1./model.nsources]*model.nsources
		M=[[0,0],[0,0]]
		r=[0]
		aux = model.aux
		S = model.sources
		for j,i in enumerate([arg1,arg2]) :
			r[0]=0.000001
			M[j][0] = eval(model_definition)
			r[0]=0.999999
			M[j][1] = eval(model_definition)
		dx = M[0][1]-M[0][0]
		dy = M[1][1]-M[1][0]
		a = dy/dx

		# now finding which sources are on the extremes for given a
		Cmax=-1e9
		Cmin=1e9
		smax_index=0
		smin_index=0
		for iS in range(model.nsources):
			sx = model.sources[arg1][iS]
			sy = model.sources[arg2][iS]
			Ctemp = sy - a*sx
			if Ctemp > Cmax :
				smax_index=iS
				Cmax=Ctemp
			if Ctemp < Cmin :
				smin_index=iS
				Cmin=Ctemp

		# setting f=0 on all sources and f=1 on the extremum to get the line for upper limit
		for i in range(len(f)):
			f[i]=0
			if i == smax_index:
				f[i]=1
		Mmax=[[0,0],[0,0]]
		for j,i in enumerate([arg1,arg2]) :
			r[0]=0.000001
			Mmax[j][0] = eval(model_definition)
			r[0]=0.999999
			Mmax[j][1] = eval(model_definition)

		axes.plot(Mmax[0],Mmax[1],'k--',alpha=0.3)


		# setting f=0 on all sources and f=1 on the extremum to get the line for lower limit
		for i in range(len(f)):
			f[i]=0
			if i == smin_index:
				f[i]=1
		Mmin=[[0,0],[0,0]]
		for j,i in enumerate([arg1,arg2]) :
			r[0]=0.000001
			Mmin[j][0] = eval(model_definition)
			r[0]=0.999999
			Mmin[j][1] = eval(model_definition)

		axes.plot(Mmin[0],Mmin[1],'k--',alpha=0.3)

		# adjusting the axes limits so the lines can be clearly visible
		axes.set_xlim(xlimlow-10,xlimhigh+10)
		axes.set_ylim(ylimlow-10,ylimhigh+10)

	def update_graph(self,model) :

		if len(self.xd2[0]) > 1:
			counter=0
			for i in range(model.nisotopes):
				for j in range(i+1,model.nisotopes):
					self.ld2[counter], = self.ax[counter].plot([self.xd2[i][-2],self.xd2[i][-1]],[self.xd2[j][-2],self.xd2[j][-1]],color='C0',alpha=0.05)
					self.ld22[counter], = self.ax[counter].plot([M[i]],[M[j]],color='C0',alpha=0.5,marker='o',markersize=0.2, linestyle='')
					#self.ax[counter].draw_artist(self.ax[counter].patch)
					self.ax[counter].draw_artist(self.ld2[counter])
					self.ax[counter].draw_artist(self.ld22[counter])
					counter+=1

		self.fig.canvas.update()
		self.fig.canvas.flush_events()


	def update(self,model,M) :

		for i in range(model.nisotopes):
			self.xd2[i].append(M[i])
		self.update_graph(model)



	def save(self,filename) :
		self.fig.savefig(filename+"path2D.png",dpi=300)
		self.fig.savefig(filename+"path2D.pdf")
		for i in range(len(self.ax)):
			self.ax[i].clear()



