import matplotlib.pyplot as plt
import matplotlib

class TimeSeriesPlot:
	def __init__(self,model):
		self.base_size=1.5
		self.fig, self.ax = plt.subplots(model.nvariables, sharex=True,figsize=(5*self.base_size,0.5*self.base_size*model.nvariables))
		self.fig.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.98, wspace=0.00, hspace=0.2)

		self.xdata = [0]
		self.ydata = [ [0] for i in range(model.nvariables)]

		self.line=[]
		self.vline=[]
		for i, a in enumerate(self.ax):
			line_temp, = a.plot(self.xdata, self.ydata[i], linestyle='-', linewidth=1, alpha=1)
			vline_temp, = a.plot([0,0], [0,1.1], linestyle='--', linewidth=1, color='black', alpha=1)
			#a.set_xlim(0,1)
			a.set_ylim(0,1)
			a.set_ylabel(model.var_list[i])
			self.line.append(line_temp)
			self.vline.append(vline_temp)
		self.ttext = self.ax[0].text(0.0,0.6, "burnout", fontsize=10)
		self.ttext.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor=None, linewidth=0))

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

	def update_graph(self,model):
		pos_text=len(self.xdata)*0.85
		if model.ii>model.burnout:
			pos_text=max(model.burnout_chain_len - 0.15*len(self.xdata),0.0)

		for i, yi in enumerate(self.ydata):
			self.line[i].set_xdata(self.xdata)
			self.line[i].set_ydata(yi)
			self.vline[i].set_xdata([model.burnout_chain_len,model.burnout_chain_len])
			self.vline[i].set_ydata([0.1,1.0])
			self.ax[i].set_xlim(0,self.xdata[-1])
			self.ax[i].set_ylim(0,1.05)
		self.ttext.set_position((pos_text,0.6));

		self.ax[i].draw_artist(self.ax[i].patch)
		self.ax[i].draw_artist(self.line[i])
		self.fig.canvas.update()
		self.fig.canvas.flush_events()


	def update(self,model,M,r,f):
		# arrays used for plotting:
		self.xdata.append(self.xdata[-1]+1)
		for i in range(model.nsources):
			self.ydata[i].append(f[i])
		for i in range(model.nauxvars):
			self.ydata[i+model.nsources].append(r[i])

		self.update_graph(model)


	def save(self,filename,fmt) :
		for ext in fmt.split(","):
			self.fig.savefig(filename+"markov_chain."+ext,dpi=300)
		for i in range(len(self.ax)):
			self.ax[i].clear()

