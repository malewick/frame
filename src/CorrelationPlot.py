import scipy.stats as st

import pandas as pd
import numpy as np
from numpy import *

import matplotlib.pyplot as plt
import matplotlib


class CorrelationPlot:

	def __init__(self,model):
		self.base_size=1.5
		# figure 2 -- the 1D histograms and correlation plots
		self.fig, self.ax = plt.subplots(model.nvariables,model.nvariables,figsize=(1.1*self.base_size*model.nvariables,self.base_size*model.nvariables))
		self.fig.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.90, wspace=0., hspace=0.)

		self.hist_line = []
		# 1D histograms
		for i in range(model.nvariables):
			hist_temp = self.ax[i,i].hist([], bins=25, density=True, range=(0,1))
			self.hist_line.append(hist_temp)
		# ----------------------------------
		self.artists_initialized=False


	def update(self,model):
		# 1D histograms
		for i in range(model.nvariables):
			self.ax[i,i].clear()
		maxy=0
		for i in range(model.nvariables):
			self.hist_line[i] = self.ax[i,i].hist(model.xd1[i], bins=25, color='C0', range=(0,1), histtype='step')
			self.ax[i,i].set_xlim(-0.1,1.15)
			self.ax[i,i].text(0.65, 0.80,model.var_list[i],color='black',size=16,family='serif',transform=self.ax[i,i].transAxes)
			maxy = max(maxy,self.ax[i,i].get_ylim()[1])
		for i in range(model.nvariables):
			self.ax[i,i].set_ylim(0.0,maxy)

		self.ax[0,0].set_ylabel("entries")

		# 2D correlations plots
		for i in range(model.nvariables-1):
			for j in range(i+1,model.nvariables):
				self.ax[i,j].clear()
				self.ax[i,j].plot(model.xd1[j],model.xd1[i],marker='o',linestyle='none',color='black', markersize=0.4, alpha=max(0.5*(1-model.chain_counter/200.),0.))
				self.ax[i,j].set_xlim(-0.1,1.15)
				self.ax[i,j].set_ylim(-0.1,1.15)
				self.plotkde(model.xd1[j],model.xd1[i],self.ax[i,j])

		# 2D correlations text
		for i in range(1,model.nvariables):
			for j in range(0,i):
				self.ax[i,j].clear()
				corr = np.corrcoef(model.xd1[i],model.xd1[j])[0,1]
				clr=""
				if corr<0 :
					clr="C0"
				elif corr>0:
					clr="C3"
				sz=abs(corr)*8.+10.
				#self.ax[i,j].text(0.3, 0.3,str("{:.2f}".format(corr)),color=clr,size=sz,usetex=True,family='serif')
				self.ax[i,j].text(0.3, 0.3,str("{:.2f}".format(corr)),color=clr,size=sz,family='serif')
				self.ax[i,j].set_xlim(-0.1,1.15)

		# setting up ticks and labels
		for i in range(model.nvariables-1):
			self.ax[i,model.nvariables-1].yaxis.tick_right()

		for j in range(0,model.nvariables):
			self.ax[0,j].xaxis.tick_top()

		for i in range(model.nvariables-1):
			for j in range(i+1,model.nvariables-1):
				#self.ax[i,j].set_xticklabels([])
				#self.ax[i,j].set_xticks([])
				self.ax[i,j].set_yticklabels([])
				self.ax[i,j].set_yticks([])

		# 2D correlations text
		for i in range(1,model.nvariables-1):
			for j in range(0,i):
				self.ax[i,j].set_xticklabels([])
				self.ax[i,j].set_xticks([])
		for i in range(1,model.nvariables):
			for j in range(0,i):
				self.ax[i,j].set_yticklabels([])
				self.ax[i,j].set_yticks([])

		for i in range(1,model.nvariables-1):
			self.ax[i,i].set_xticklabels([])
			self.ax[i,i].set_xticks([])
		for i in range(model.nvariables):
			yticks = self.ax[i,i].yaxis.get_major_ticks()
			yticks[0].label1.set_visible(False)

		if self.artists_initialized:
			for i in range(0,model.nvariables):
				for j in range(0,model.nvariables):
					self.ax[i,j].draw_artist(self.ax[i,j].patch)
					for line in self.ax[i,j].lines :
						self.ax[i,j].draw_artist(line)
			self.fig.canvas.update()
			self.fig.canvas.flush_events()
		else :
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
			self.artists_initialized=True


	def plotkde(self, x,y,ax) :
		if len(x)<10:
			return
		xoffset = 0.1*(max(x)-min(x))
		yoffset = 0.1*(max(y)-min(y))
		xmin, xmax = min(x)-xoffset, max(x)+xoffset
		ymin, ymax = min(y)-yoffset, max(y)+yoffset

		# Peform the kernel density estimate
		xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([xx.ravel(), yy.ravel()])
		values = np.vstack([x, y])
		if np.isnan(values).any():
			return
		kernel = st.gaussian_kde(values)
		f = np.reshape(kernel(positions).T, xx.shape)

		# Contourf plot
		#cfset = ax.contourf(xx, yy, f, cmap='Blues')
		## Or kernel density estimate plot instead of the contourf plot
		#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
		# Contour plot
		cset = ax.contour(xx, yy, f, cmap='Blues')
		# Label plot
		#ax.clabel(cset, inline=1, fontsize=10)


	def save(self,filename,fmt):

		for ext in fmt.split(","):
			self.fig.savefig(filename+"correlations."+ext,dpi=300)

		for i in range(len(self.ax)):
			for j in range(len(self.ax[i])):
				self.ax[i,j].clear()


