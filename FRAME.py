import sys
import time
import matplotlib
matplotlib.use('Qt5Agg')

from PySide2.QtCore import Qt, QThread
from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


import pandas as pd
import numpy as np
from numpy import *

import sympy

from multiprocessing import Process, Queue


import NDimModel

import xml.etree.ElementTree as ET
from xml.dom import minidom

import re


class ModelWrapper(QThread, NDimModel.Model) :
	"""
	Runs a separate thread for the model so it can emit signals and UI can listen.
	"""
	countChanged = QtCore.Signal(int)
	def __init__(self):
		QThread.__init__(self)
		NDimModel.Model.__init__(self)

	def run(self) :
		self.countChanged.emit(int(100*(self.chain_counter / self.max_chain_entries)))
		self.run_model()


class Worker(QtCore.QObject, NDimModel.Model) :

	def __init__(self):
		QtCore.QObject.__init__(self)
		NDimModel.Model.__init__(self)
		self.ready=False

	@QtCore.Slot()
	def process(self):
		self.run_model()
		self.ready=True
	
	


class ExternalThread(QThread) :
	countChanged = QtCore.Signal(int)

	def __init__(self,model):
		self.model = model
		QThread.__init__(self)

	def run(self):
		while True:
			time.sleep(1)
			progress = self.model.group_i / self.model.ngroups + 1./self.model.ngroups * min(1., self.model.chain_counter / self.model.max_chain_entries)
			if self.model.is_finished :
				progress=1
			self.countChanged.emit(int(100*progress))


#class EmittingStream(QtCore.QObject):
class EmittingStream(QThread):

	textWritten = QtCore.Signal(str)

	def write(self, text):
		self.textWritten.emit(str(text))


class Log(object):
	def __init__(self, edit):
		self.out = sys.stdout
		self.textEdit = edit

	def write(self, message):
		self.out.write(message)
		#self.textEdit.append(message)

		cursor = self.textEdit.textCursor()
		cursor.movePosition(QtGui.QTextCursor.End)
		cursor.insertText(message)
		self.textEdit.setTextCursor(cursor)
		self.textEdit.ensureCursorVisible()

	def flush(self):
		self.out.flush()



class MplCanvas(FigureCanvasQTAgg):


	def __init__(self, parent=None, width=6, height=4, dpi=300):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.fig.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.15)
		super(MplCanvas, self).__init__(self.fig)

	def plot_data(self, df, arg1, arg2):
		for i, group in enumerate(df['group'].unique()):
			dfs=df.loc[df['group']==group]
			print(dfs[arg1].tolist(), dfs[arg2].tolist())
			#self.axes.scatter(dfs[arg1].tolist(), dfs[arg2].tolist(),
			#				  edgecolor="C"+str(i%10),
			#				  facecolor="C"+str(i%10),
			#				  #facecolor="None",
			#				  #marker="$"+str(group)+"$",
			#				  marker='+',
			#				  alpha=1.,
			#)
			#for sample in dfs['sample_id'].to_list():
			#	self.axes.annotate(str(sample),(df[arg1].iloc[sample],df[arg2].iloc[sample]),
			#				size=8,
			#				color="C"+str(i%10))

			for sample in dfs['sample_id'].to_list():
				size=30
				if sample>9:
					size=75
				self.axes.scatter(df[arg1].iloc[sample], df[arg2].iloc[sample],
								  facecolor="C"+str(i%10),
								  edgecolor="None",
								  marker="$"+str(sample)+"$",
								  s=size,
								  alpha=1.,
				)

		self.axes.set_xlabel(arg1)
		self.axes.set_ylabel(arg2)

			
		self.figure.canvas.draw()



	def plot_sources(self, df_model, arg1, arg2):

		# plotting the sources with delta as rectangles 
		for j, tstr in enumerate(df_model["source"].unique()) :
			vx=df_model.loc[(df_model['source'] == tstr)].iloc[0][arg1]
			dx=df_model.loc[(df_model['source'] == tstr)].iloc[0]['delta('+arg1+')']
			sx=df_model.loc[(df_model['source'] == tstr)].iloc[0]['sigma('+arg1+')']
			vy=df_model.loc[(df_model['source'] == tstr)].iloc[0][arg2]
			dy=df_model.loc[(df_model['source'] == tstr)].iloc[0]['delta('+arg2+')']
			sy=df_model.loc[(df_model['source'] == tstr)].iloc[0]['sigma('+arg2+')']
			print(vx,vy,dx,dy,sx,sy)
			self.axes.errorbar(vx,vy,
					       xerr=sx,
					       yerr=sy,
					       ecolor="black",
					       mec="None",
					       mfc="None",
					       marker="o",
					       linestyle='',
					       alpha=1.,
					)
			rect = Rectangle((vx-dx,vy-dy),2*dx,2*dy,linewidth=1,edgecolor='black',facecolor="black",alpha=0.15)
			self.axes.add_patch(rect)
			offx=max(sx,dx)
			offy=max(sy,dy)
			self.axes.text(vx-offx+0.5,vy-offy+0.5, tstr, fontsize=12)

		self.axes.autoscale()
		self.figure.canvas.draw()


	def plot_fractionation(self, model, arg1, arg2):

		# shows linescorresponding to the fractionation process
		# makes sense only with one auxiliary variable
		if model.nauxvars != 1:
			return

		# in order to overwrite autoscale
		xlimlow = self.axes.get_xlim()[0]
		xlimhigh = self.axes.get_xlim()[1]
		ylimlow = self.axes.get_ylim()[0]
		ylimhigh = self.axes.get_ylim()[1]

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
		
		self.axes.plot(Mmax[0],Mmax[1],'k--',alpha=0.3)


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
		
		self.axes.plot(Mmin[0],Mmin[1],'k--',alpha=0.3)

		# adjusting the axes limits so the lines can be clearly visible
		self.axes.set_xlim(xlimlow-10,xlimhigh+10)
		self.axes.set_ylim(ylimlow-10,ylimhigh+10)

		self.figure.canvas.draw()





class TableModel(QtCore.QAbstractTableModel):
	def __init__(self, data):
		super(TableModel, self).__init__()
		self._data = data
		self.header_labels=data.columns

	def data(self, index, role):
		if role == Qt.DisplayRole:
			# See below for the nested-list data structure.
			# .row() indexes into the outer list,
			# .column() indexes into the sub-list
			#return self._data[index.row()][index.column()]
			#print(index.row(), index.column(), self._data.iat[index.row(),index.column()])
			return str(self._data.iat[index.row(),index.column()])

	def rowCount(self, index):
		# The length of the outer list.
		return len(self._data.index)

	def columnCount(self, index):
		# The following takes the first sub-list, and returns
		# the length (only works if all rows are an equal length)
		return len(self._data.columns)

	def headerData(self, section, orientation, role=Qt.DisplayRole):
		if role == Qt.DisplayRole and orientation == Qt.Horizontal:
			return self.header_labels[section]

	def reset(self) :
		self.beginResetModel()
		self._data=pd.DataFrame(data={'-':[0]})
		self.header_labels=self._data.columns
		self.endResetModel()



class MainWindow(QtWidgets.QMainWindow):

 

	def __init__(self, *args, **kwargs):
		super(MainWindow, self).__init__(*args, **kwargs)

		self.setWindowTitle("FRAME")

		self.df_data = pd.DataFrame(data={'-':[0]})
		self.df_srcs = pd.DataFrame(data={'-':[0]})
		self.df_auxs = pd.DataFrame(data={'-':[0]})

		#self.mcmc_model = ModelWrapper()
		self.mcmc_model = Worker()

		self.ndim=0


		self.table1 = QtWidgets.QTableView()
		self.model1 = TableModel(self.df_data)
		self.table1.setModel(self.model1)
		self.table1.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents);

		self.table2 = QtWidgets.QTableView()
		self.model2 = TableModel(self.df_srcs)
		self.table2.setModel(self.model2)
		self.table2.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents);

		self.table3 = QtWidgets.QTableView()
		self.model3 = TableModel(self.df_auxs)
		self.table3.setModel(self.model3)
		self.table3.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents);

		self.outdir = "./output/"

		self.sc1 = MplCanvas(self, width=2.5, height=3, dpi=100)

		self.toolbar1 = NavigationToolbar(self.sc1, self)

		self.sc2 = MplCanvas(self, width=2.5, height=3, dpi=100)

		self.toolbar2 = NavigationToolbar(self.sc2, self)

		self.sc3 = MplCanvas(self, width=2.5, height=3, dpi=100)

		self.toolbar3 = NavigationToolbar(self.sc3, self)

		placeholder = QtWidgets.QVBoxLayout()
		plchldr_lbl = QtWidgets.QLabel("")
		placeholder.addWidget(plchldr_lbl)

		placeholder2 = QtWidgets.QVBoxLayout()
		plchldr_lbl2 = QtWidgets.QLabel("")
		placeholder2.addWidget(plchldr_lbl2)

		# handling layouts
		layout1 = QtWidgets.QVBoxLayout()
		layout1.addWidget(self.toolbar1)
		layout1.addWidget(self.sc1)

		layout2 = QtWidgets.QVBoxLayout()
		layout2.addWidget(self.toolbar2)
		layout2.addWidget(self.sc2)

		layout3 = QtWidgets.QVBoxLayout()
		layout3.addWidget(self.toolbar3)
		layout3.addWidget(self.sc3)

		self.toolbar1.hide()
		self.toolbar2.hide()
		self.toolbar3.hide()

		button_load = QtWidgets.QPushButton("1. Load samples\n")
		button_sources = QtWidgets.QPushButton("2. Load sources\n")
		button_frac = QtWidgets.QPushButton("3. Load aux. data \n (optional)")

		button_load.setSizePolicy(
			QtWidgets.QSizePolicy.Preferred,
			QtWidgets.QSizePolicy.Expanding)
		button_load.setFont(QtGui.QFont('Sans', 15)) 

		button_sources.setSizePolicy(
			QtWidgets.QSizePolicy.Preferred,
			QtWidgets.QSizePolicy.Expanding)
		button_sources.setFont(QtGui.QFont('Sans', 15)) 

		button_frac.setSizePolicy(
			QtWidgets.QSizePolicy.Preferred,
			QtWidgets.QSizePolicy.Expanding)
		button_frac.setFont(QtGui.QFont('Sans', 15)) 

		button_load.clicked.connect(self.load_data_clicked)
		button_sources.clicked.connect(self.load_sources_clicked)
		button_frac.clicked.connect(self.load_aux_clicked)


		self.layout_dropdown = QtWidgets.QHBoxLayout();
		self.iso_qlabels=[]
		self.iso_qlabels.append(QtWidgets.QLabel("-"))
		for ql in self.iso_qlabels:
			self.layout_dropdown.addWidget(ql)

		gb = QtWidgets.QGroupBox("Isotopes");
		gb.setFont(QtGui.QFont('Sans', 11)) 
		gb.setLayout(self.layout_dropdown)

		self.layout_dropdown2 = QtWidgets.QHBoxLayout();
		self.src_qlabels=[]
		self.src_qlabels.append(QtWidgets.QLabel("src"))
		for ql in self.src_qlabels:
			self.layout_dropdown2.addWidget(ql)

		gbs = QtWidgets.QGroupBox("Sources");
		gbs.setLayout(self.layout_dropdown2)
		gbs.setFont(QtGui.QFont('Sans', 11)) 

		self.layout_dropdown1 = QtWidgets.QHBoxLayout();
		self.auxvar_qlabels=[]
		self.auxvar_qlabels.append(QtWidgets.QLabel("-"))
		for ql in self.auxvar_qlabels:
			self.layout_dropdown1.addWidget(ql)

		gbv = QtWidgets.QGroupBox("Aux. variables");
		gbv.setFont(QtGui.QFont('Sans', 11)) 
		gbv.setLayout(self.layout_dropdown1)

		self.layout_dropdown3 = QtWidgets.QHBoxLayout();
		self.aux_qlabels=[]
		self.aux_qlabels.append(QtWidgets.QLabel("-"))
		for ql in self.aux_qlabels:
			self.layout_dropdown3.addWidget(ql)

		gba = QtWidgets.QGroupBox("Aux. parameters");
		gba.setFont(QtGui.QFont('Sans', 11)) 
		gba.setLayout(self.layout_dropdown3)


		#layout_plotting = QtWidgets.QFormLayout();
		#lp1 = QtWidgets.QLabel("markers")
		#cbp1 = QtWidgets.QComboBox()
		#cbp1.addItem("group numbers")
		#cbp1.addItem("circles")
		#cbp1.addItem("squares")
		#lp2 = QtWidgets.QLabel("toolbar")
		#bp = QtWidgets.QPushButton("Show/Hide", checkable=True)
		#layout_plotting.addRow(lp1, cbp1)
		#layout_plotting.addRow(lp2, bp)

		#gb2 = QtWidgets.QGroupBox("Plotting");
		#gb2.setLayout(layout_plotting)
		#layout_aux = QtWidgets.QHBoxLayout()
		#layout_aux.addWidget(gb2)

		layout_buttons1 = QtWidgets.QVBoxLayout()
		layout_buttons1.addWidget(button_load)
		#layout_buttons1.addStretch(1)

		layout_buttons2 = QtWidgets.QVBoxLayout()
		layout_buttons2.addWidget(button_sources)
		#layout_buttons2.addStretch(1)

		layout_buttons3 = QtWidgets.QVBoxLayout()
		layout_buttons3.addWidget(button_frac)
		#layout_buttons3.addStretch(1)


		layout_allothers = QtWidgets.QVBoxLayout()
		layout_allothers.addWidget(gb)
		layout_allothers.addWidget(gbs)
		layout_allothers.addWidget(gbv)
		layout_allothers.addWidget(gba)
		#layout_allothers.addLayout(layout_aux)



	# -----
	# ----- Second card
	# -----

		label1 = QtWidgets.QLabel("Number of iterations: ");
		label2 = QtWidgets.QLabel("Burn-out iterations: ");
		label3 = QtWidgets.QLabel("Desired Markov-chain length: ");
		
		self.lineEdit1 = QtWidgets.QLineEdit("10000");
		self.lineEdit2 = QtWidgets.QLineEdit("1000");
		self.lineEdit3 = QtWidgets.QLineEdit("300");
		
		formLayout = QtWidgets.QFormLayout();
		formLayout.addRow(label1, self.lineEdit1)
		formLayout.addRow(label2, self.lineEdit2)
		formLayout.addRow(label3, self.lineEdit3)

		label4 = QtWidgets.QLabel("Output directory: ");
		self.lineEdit4 = QtWidgets.QLineEdit("./output/");
		button4 = QtWidgets.QPushButton("Set...")
		button4.clicked.connect(self.set_outdir)
		formLayout2 = QtWidgets.QFormLayout();
		formLayout2.addRow(label4, self.lineEdit4)
		layout_outdir = QtWidgets.QHBoxLayout()
		layout_outdir.addLayout(formLayout2)
		layout_outdir.addWidget(button4)
		#layout_outdir.addStretch(1)

		self.plotting_check_box = QtWidgets.QPushButton("Batch mode: OFF", self);
		self.plotting_check_box.setCheckable(True);
		self.plotting_check_box.setToolTip("Batch mode turns off online plotting, which affects the computation time significantly.")
		self.plotting_check_box.setFont(QtGui.QFont('Sans', 12)) 
		self.plotting_check_box.clicked.connect(self.on_off_swtich)

		self.button_plot_settings = QtWidgets.QPushButton("Plot settings...")
		self.button_plot_settings.setFont(QtGui.QFont('Sans', 11)) 
		self.button_plot_settings.clicked.connect(self.show_plot_settings)
		self.button_plot_settings.setToolTip("Set nice labels for axes using LaTex format.")

		layout_plotting_buttons = QtWidgets.QHBoxLayout()
		layout_plotting_buttons.addWidget(self.plotting_check_box)
		layout_plotting_buttons.addWidget(self.button_plot_settings)

		layout_aux = QtWidgets.QVBoxLayout()
		label_model = QtWidgets.QLabel("Model equation:")
		label_model.setFont(QtGui.QFont('Sans', 11)) 
		label_derivatives = QtWidgets.QLabel("Model derivatives:")
		label_derivatives.setFont(QtGui.QFont('Sans', 11)) 
		self.ledit_model = QtWidgets.QLineEdit("");
		self.ledit_model.setMinimumSize(200,25)
		self.ledit_model.setReadOnly(True)
		self.ledit_derivatives = QtWidgets.QLineEdit("");
		self.ledit_derivatives.setMinimumSize(200,25)
		self.ledit_derivatives.setReadOnly(True)
		layout_aux.addWidget(label_model)
		layout_aux.addWidget(self.ledit_model)
		layout_aux.addWidget(label_derivatives)
		layout_aux.addWidget(self.ledit_derivatives)
		layout_aux.addStretch(1)


		layout_form_outdir = QtWidgets.QVBoxLayout()
		layout_form_outdir.addLayout(formLayout)
		layout_form_outdir.addLayout(layout_outdir)
		layout_form_outdir.addLayout(layout_plotting_buttons);
		layout_form_outdir.addLayout(layout_aux)

		button1 = QtWidgets.QPushButton("4. Run model\n")
		button1.setSizePolicy(
			QtWidgets.QSizePolicy.Preferred,
			QtWidgets.QSizePolicy.Expanding)
		button1.setFont(QtGui.QFont('Sans', 15)) 
	
		layout_buttons = QtWidgets.QHBoxLayout()
		layout_buttons.addWidget(button1)
		#layout_buttons.addStretch(1)

		self.progressbar = QtWidgets.QProgressBar()
		self.progressbar.setMaximum(100)

		self.abort_button = QtWidgets.QPushButton("Stop")
		self.abort_button.setFont(QtGui.QFont('Sans', 15)) 
		self.abort_button.clicked.connect(self.abort_model)

		layout_progress = QtWidgets.QHBoxLayout()
		layout_progress.addWidget(self.progressbar)
		layout_progress.addWidget(self.abort_button)






		layout_data = QtWidgets.QGridLayout()
		layout_data.addLayout(layout_buttons1,0,0,1,1)
		layout_data.addLayout(layout_buttons2,0,1,1,1)
		layout_data.addLayout(layout_buttons3,0,2,1,1)
		layout_data.addLayout(layout_buttons,0,3,1,1)
		layout_data.addWidget(self.table1,1,0)
		layout_data.addWidget(self.table2,1,1)
		layout_data.addWidget(self.table3,1,2)
		layout_data.addLayout(layout_form_outdir,1,3)
		layout_data.setRowStretch(0,1)
		layout_data.setRowStretch(1,4)


		layout_data2 = QtWidgets.QGridLayout()
		layout_data2.addLayout(layout_data,0,0,1,4)
		layout_data2.addLayout(layout1,1,0)
		layout_data2.addLayout(layout2,1,1)
		layout_data2.addLayout(layout3,1,2)
		layout_data2.addLayout(layout_allothers,1,3)



		self.logConsole = QtWidgets.QTextEdit()
		self.logConsole.setReadOnly(True)

		layout_model = QtWidgets.QVBoxLayout()
		layout_model.addLayout(layout_progress)
		layout_model.addWidget(self.logConsole)

		self.tabs = QtWidgets.QTabWidget()

		widget_data = QtWidgets.QWidget()
		widget_data.setLayout(layout_data2)
		self.tabs.addTab(widget_data, "Input Data, Model")

		widget_model = QtWidgets.QWidget()
		widget_model.setLayout(layout_model)
		self.tabs.addTab(widget_model, "Running MCMC")

		button1.clicked.connect(self.run_model)








		# Create a placeholder widget to hold our toolbar and canvas.
		#widget = QtWidgets.QWidget()
		#widget.setLayout(layout)
		#self.setCentralWidget(widget)
		self.setCentralWidget(self.tabs)

		menuBar = self.menuBar()
		fileMenu = menuBar.addMenu("&File");
		viewMenu = menuBar.addMenu("&View");

		action_new = QtWidgets.QAction("&New", self);
		action_new.setStatusTip("Start a new analysis from scratch")
		action_new.triggered.connect(self.clear_clicked)
		fileMenu.addAction(action_new);

		action_save = QtWidgets.QAction("&Save", self);
		action_save.setStatusTip("Save session")
		action_save.triggered.connect(self.save_xml_clicked)
		fileMenu.addAction(action_save);

		action_load = QtWidgets.QAction("&Load", self);
		action_load.setStatusTip("Load session")
		action_load.triggered.connect(self.load_xml_clicked)
		fileMenu.addAction(action_load);

		action_pref = QtWidgets.QAction("&Preferences", self);
		action_pref.setStatusTip("Preferences and Settings")
		action_pref.triggered.connect(self.show_preferences)
		fileMenu.addAction(action_pref);


		self.action_mpl_toolbar = QtWidgets.QAction("&Plotting toolbar", self);
		self.action_mpl_toolbar.setCheckable(True);
		self.action_mpl_toolbar.setChecked(False);
		self.action_mpl_toolbar.setStatusTip("Show toolbars for plotting")
		self.action_mpl_toolbar.triggered.connect(self.show_hide_toolbar)
		viewMenu.addAction(self.action_mpl_toolbar);

		sys.stdout = Log(self.logConsole)

		#es = EmittingStream()
		#sys.stdout = es
		#self.connect(sys.stdout,QtCore.SIGNAL('textWritten(QString)'),self.normalOutputWritten)
		#sys.stdout = EmittingStream()
		#self.connect(sys.stdout,QtCore.SIGNAL('textWritten(QString)'),self.write2Console)




		self.show()


	def __del__(self):
		# Restore sys.stdout
		sys.stdout = sys.__stdout__

	def onCountChanged(self, value):
		self.progressbar.setValue(value)


	def normalOutputWritten(self, text):
		"""Append text to the QTextEdit."""
		# Maybe QTextEdit.append() works as well, but this is how I do it:
		cursor = self.logConsole.textCursor()
		cursor.movePosition(QtGui.QTextCursor.End)
		cursor.insertText(text)
		self.logConsole.setTextCursor(cursor)
		self.logConsole.ensureCursorVisible()


	def show_hide_toolbar(self):
		if self.action_mpl_toolbar.isChecked():
			self.toolbar1.show()
			self.toolbar2.show()
			self.toolbar3.show()
		else:
			self.toolbar1.hide()
			self.toolbar2.hide()
			self.toolbar3.hide()

	def on_off_swtich(self):
		if self.plotting_check_box.isChecked():
			self.plotting_check_box.setStyleSheet("background-color : skyblue")
			self.plotting_check_box.setText("Batch mode: ON ");
			self.mcmc_model.plotting_switch=True
		else:
			self.plotting_check_box.setStyleSheet("background-color : lightgrey")
			self.plotting_check_box.setText("Batch mode: OFF");
			self.mcmc_model.plotting_switch=False

	def show_preferences(self):
		print("showing preferences")
		self.preferences_window = QtWidgets.QWidget()
		self.preferences_window.setWindowTitle("Preferences")
		layout = QtWidgets.QVBoxLayout()
		formLayout = QtWidgets.QFormLayout()
		label00 = QtWidgets.QLabel("property")
		label01 = QtWidgets.QLabel("value")
		formLayout.addRow(label00,label01)

		label_1=QtWidgets.QLabel("default delimiter")
		self.lineEdit_1=QtWidgets.QLineEdit()
		self.lineEdit_1.setText(self.mcmc_model.default_delimiter)
		formLayout.addRow(label_1,self.lineEdit_1)

		line1 = QtWidgets.QFrame()
		line1.setFrameShape(QtWidgets.QFrame.HLine)
		#line1.setFrameShadow(QtWidgets.QFrame.Sunken)

		formLayout2 = QtWidgets.QFormLayout()

		vlayout = QtWidgets.QVBoxLayout()
		label10 = QtWidgets.QLabel("You can use following variables:")
		label11 = QtWidgets.QLabel("""'data_file', 'sources_file', 'aux_file'""")
		vlayout.addWidget(label10)
		vlayout.addWidget(label11)

		label_2=QtWidgets.QLabel("output file name (.csv)")
		self.lineEdit_2=QtWidgets.QLineEdit()
		self.lineEdit_2.setText(""""results_"+data_file+"_"+sources_file+"_"+aux_file""")
		# "
		formLayout2.addRow(label_2,self.lineEdit_2)

		vlayout2 = QtWidgets.QVBoxLayout()
		label20 = QtWidgets.QLabel("You can use following variables:")
		label21 = QtWidgets.QLabel("""'data_file', 'sources_file', 'aux_file', 'group', 'plot_title'""")
		vlayout2.addWidget(label20)
		vlayout2.addWidget(label21)

		formLayout3 = QtWidgets.QFormLayout()

		label_3=QtWidgets.QLabel("plot file names (.pdf, .png)")
		self.lineEdit_3=QtWidgets.QLineEdit()
		self.lineEdit_3.setText("""plot_title+"_"+data_file+"_"+sources_file+"_"+aux_file++"_group"+group""")
		# "
		formLayout3.addRow(label_3,self.lineEdit_3)

		hlayout = QtWidgets.QHBoxLayout()
		cancel_button = QtWidgets.QPushButton("Cancel")
		cancel_button.clicked.connect(self.close_preferences)
		ok_button = QtWidgets.QPushButton("OK")
		ok_button.clicked.connect(self.accept_preferences)
		hlayout.addWidget(cancel_button)
		hlayout.addWidget(ok_button)

		layout.addLayout(formLayout)
		layout.addWidget(line1)
		layout.addLayout(vlayout)
		layout.addLayout(formLayout2)
		layout.addLayout(vlayout2)
		layout.addLayout(formLayout3)
		layout.addLayout(hlayout)
		self.preferences_window.setLayout(layout)
		self.preferences_window.show()

	def close_preferences(self):
		print("Closing preferences...")
		self.preferences_window.close()

	def accept_preferences(self):
		print("Accept and close preferences...")
		self.mcmc_model.default_delimiter = self.lineEdit_1.text()
		self.preferences_window.close()


	def show_plot_settings(self):
		print("show plot settings")
		self.settings_window = QtWidgets.QWidget()
		self.settings_window.setWindowTitle("Plot Settings")
		layout = QtWidgets.QVBoxLayout()
		formLayout = QtWidgets.QFormLayout()
		label00 = QtWidgets.QLabel("isotope")
		label01 = QtWidgets.QLabel("axis label")
		formLayout.addRow(label00,label01)
		self.latex_labels=[]
		for i in range(self.ndim):
			iso = self.mcmc_model.isotopes_list[i]
			tmp_label=QtWidgets.QLabel(iso)
			tmp_lineEdit=QtWidgets.QLineEdit()
			latex_iso = u'$' + re.sub(r'd', r'\\delta', iso)
			latex_iso = re.sub(r'(\d{1,3})',r'^{\1}', latex_iso)
			latex_iso += u'$ [‰]'
			self.latex_labels.append(latex_iso)
			tmp_lineEdit.setText(latex_iso)
			formLayout.addRow(tmp_label,tmp_lineEdit)
		hlayout = QtWidgets.QHBoxLayout()
		cancel_button = QtWidgets.QPushButton("Cancel")
		cancel_button.clicked.connect(self.close_plot_settings)
		ok_button = QtWidgets.QPushButton("OK")
		ok_button.clicked.connect(self.accept_plot_settings)
		hlayout.addWidget(cancel_button)
		hlayout.addWidget(ok_button)

		layout.addLayout(formLayout)
		layout.addLayout(hlayout)
		self.settings_window.setLayout(layout)
		self.settings_window.show()

	def close_plot_settings(self):
		print("Closing plotting settings...")
		self.settings_window.close()

	def accept_plot_settings(self):
		print("Accept and close plotting settings...")
		self.mcmc_model.axes_labels = self.latex_labels.copy()

		if self.ndim==2 :
			self.sc1.axes.set_xlabel(self.latex_labels[0])
			self.sc1.axes.set_ylabel(self.latex_labels[1])
			self.sc1.figure.canvas.draw()
		if self.ndim==3 :
			self.sc1.axes.set_xlabel(self.latex_labels[0])
			self.sc1.axes.set_ylabel(self.latex_labels[1])
			self.sc2.axes.set_xlabel(self.latex_labels[0])
			self.sc2.axes.set_ylabel(self.latex_labels[2])
			self.sc3.axes.set_xlabel(self.latex_labels[1])
			self.sc3.axes.set_ylabel(self.latex_labels[2])
			self.sc1.figure.canvas.draw()
			self.sc2.figure.canvas.draw()
			self.sc3.figure.canvas.draw()

		self.settings_window.close()



	def clear_clicked(self) :
		self.model1.reset()
		self.model2.reset()
		self.model3.reset()

		self.sc1.axes.clear()
		self.sc2.axes.clear()
		self.sc3.axes.clear()
		self.sc1.figure.canvas.draw()
		self.sc2.figure.canvas.draw()
		self.sc3.figure.canvas.draw()


	def save_xml_clicked(self) :
		print("Saving xml file")
		fileName = QtWidgets.QFileDialog.getSaveFileName(self,
				"Save File", "./input/", "XML Files *.xml")
		print(fileName)
		self.save_to_xml(fileName[0])

	def save_to_xml(self, xml_file):

		if not hasattr(self,"data_file") or not hasattr(self,"sources_file") or self.data_file==None or self.sources_file==None:
			print("Nothing to save. Please load data and sources first.")
			return

		# create the file structure
		data = ET.Element('data')
		items = ET.SubElement(data, 'items')
		item1 = ET.SubElement(items, 'item')
		item2 = ET.SubElement(items, 'item')
		item1.set('name','data_file')
		item2.set('name','sources_file')
		item1.text = self.data_file
		item2.text = self.sources_file
		if hasattr(self,"aux_file") and self.aux_file != None:
			item3 = ET.SubElement(items, 'item')
			item3.set('name','aux_file')
			item3.text = self.aux_file

		xmlstr = minidom.parseString(ET.tostring(data)).toprettyxml(indent="\t")
		with open(xml_file, 'w') as f:
			f.write(xmlstr)

		

	def load_xml_clicked(self) :
		print("Loading from xml file")
		fileName = QtWidgets.QFileDialog.getOpenFileName(self,
				"Open File", "./input/", "XML Files *.xml")
		print(fileName[0])
		self.load_from_xml(fileName[0])
		

	def load_from_xml(self, xml_file):
		tree = ET.parse(xml_file)
		root = tree.getroot()
		dict_from_xml={}
		for elem in root:
			for subelem in elem:
				print(subelem.attrib, subelem.text)
				dict_from_xml[subelem.attrib['name']] = subelem.text
		print(dict_from_xml)

		self.mcmc_model.initialized=False
		self.load_data(dict_from_xml['data_file'])
		self.load_sources(dict_from_xml['sources_file'])
		if 'aux_file' in dict_from_xml.keys():
			self.load_aux(dict_from_xml['aux_file'])


	

	def load_data_clicked(self) :
		print("Loading data on measurements...")
		fileName = QtWidgets.QFileDialog.getOpenFileName(self,
				"Open File", "./input/", "CSV Files *.csv")
		print(fileName[0])
		self.load_data(fileName[0])


	def load_data(self,filename) :

		self.data_file=filename
		self.mcmc_model.load_measurements(filename)

		self.df_data = self.mcmc_model.df

		print(self.df_data)

		self.model1 = TableModel(self.df_data)
		self.table1.setModel(self.model1)

		# clearing layout
		for i in reversed(range(self.layout_dropdown.count())): 
			widgetToRemove = self.layout_dropdown.itemAt(i).widget()
			self.layout_dropdown.removeWidget(widgetToRemove)
			widgetToRemove.setParent(None)

		self.var_qlabels=[]
		for v in self.mcmc_model.isotopes_list:
			self.var_qlabels.append(QtWidgets.QLabel(v))
		for ql in self.var_qlabels:
			self.layout_dropdown.addWidget(ql)

		self.ndim = len(self.mcmc_model.isotopes_list)

		if self.ndim==2 :
			self.sc1.plot_data(self.df_data,self.mcmc_model.isotopes_list[0],self.mcmc_model.isotopes_list[1])
		if self.ndim==3 :
			self.sc1.plot_data(self.df_data,self.mcmc_model.isotopes_list[0],self.mcmc_model.isotopes_list[1])
			self.sc2.plot_data(self.df_data,self.mcmc_model.isotopes_list[0],self.mcmc_model.isotopes_list[2])
			self.sc3.plot_data(self.df_data,self.mcmc_model.isotopes_list[1],self.mcmc_model.isotopes_list[2])

		
	def load_sources_clicked(self) :
		print("Loading data on sources...")
		fileName = QtWidgets.QFileDialog.getOpenFileName(self,
				"Open File", "./input/", "CSV Files *.csv")
		self.load_sources(fileName[0])

	def load_sources(self, filename) :

		self.sources_file=filename
		self.mcmc_model.load_sources(filename)

		self.df_srcs = self.mcmc_model.df_sources

		self.model2 = TableModel(self.df_srcs)
		self.table2.setModel(self.model2)

		# clearing layout
		for i in reversed(range(self.layout_dropdown2.count())): 
			widgetToRemove = self.layout_dropdown2.itemAt(i).widget()
			self.layout_dropdown2.removeWidget(widgetToRemove)
			widgetToRemove.setParent(None)

		self.src_qlabels=[]
		for src in self.mcmc_model.sources_list:
			self.src_qlabels.append(QtWidgets.QLabel(src))
		for ql in self.src_qlabels:
			self.layout_dropdown2.addWidget(ql)

		if self.ndim==2 :
			self.sc1.plot_sources(self.df_srcs,self.mcmc_model.isotopes_list[0],self.mcmc_model.isotopes_list[1])
		if self.ndim==3 :
			self.sc1.plot_sources(self.df_srcs,self.mcmc_model.isotopes_list[0],self.mcmc_model.isotopes_list[1])
			self.sc2.plot_sources(self.df_srcs,self.mcmc_model.isotopes_list[0],self.mcmc_model.isotopes_list[2])
			self.sc3.plot_sources(self.df_srcs,self.mcmc_model.isotopes_list[1],self.mcmc_model.isotopes_list[2])

		self.ledit_model.setText("M[i] = M0[i]")

		derivatives_string=""
		for i, src in enumerate(self.mcmc_model.sources_list):
			derivatives_string += "dM/d"+src+"="+"f["+str(i)+"]"+", "
		self.ledit_derivatives.setText(derivatives_string[:-2])


	def load_aux_clicked(self) :
		print("Loading aux data...")
		fileName = QtWidgets.QFileDialog.getOpenFileName(self,
				"Open File", "./input/", "CSV Files *.csv")

		self.load_aux(fileName[0])


	def load_aux(self, filename) :

		self.aux_file=filename
		self.mcmc_model.load_aux(filename)
		self.mcmc_model.set_up_data()

		self.ledit_model.setText("M = " + self.mcmc_model.model_definition);
		str_derivaties=""
		for i, src in enumerate(self.mcmc_model.sources_list):
			str_derivaties += "dM/d"+src+"="+self.mcmc_model.model_derivatives[i]+", "
		for i, par in enumerate(self.mcmc_model.aux_pars):
			str_derivaties += "dM/d"+par+"="+self.mcmc_model.model_derivatives[len(self.mcmc_model.sources_list)+i]+", "
			
		self.ledit_derivatives.setText(str_derivaties[:-2]);

		self.df_auxs = self.mcmc_model.df_aux

		self.model3 = TableModel(self.df_auxs)
		self.table3.setModel(self.model3)

		# clearing layout
		for i in reversed(range(self.layout_dropdown1.count())): 
			widgetToRemove = self.layout_dropdown1.itemAt(i).widget()
			self.layout_dropdown1.removeWidget(widgetToRemove)
			widgetToRemove.setParent(None)

		self.auxvar_qlabels=[]
		for v in self.mcmc_model.aux_vars:
			self.auxvar_qlabels.append(QtWidgets.QLabel(v))
		for ql in self.auxvar_qlabels:
			self.layout_dropdown1.addWidget(ql)

		# clearing layout
		for i in reversed(range(self.layout_dropdown3.count())): 
			widgetToRemove = self.layout_dropdown3.itemAt(i).widget()
			self.layout_dropdown3.removeWidget(widgetToRemove)
			widgetToRemove.setParent(None)

		self.aux_qlabels=[]
		for p in self.mcmc_model.aux_pars:
			self.aux_qlabels.append(QtWidgets.QLabel(p))
		for ql in self.aux_qlabels:
			self.layout_dropdown3.addWidget(ql)

		if self.ndim==2 :
			self.sc1.plot_fractionation(self.mcmc_model,0,1)
		if self.ndim==3 :
			self.sc1.plot_fractionation(self.mcmc_model,0,1)
			self.sc2.plot_fractionation(self.mcmc_model,0,2)
			self.sc3.plot_fractionation(self.mcmc_model,1,2)


	def set_outdir(self) :
		print("Choosing output directory...")
		dirName = QtWidgets.QFileDialog.getExistingDirectory( self,
				'Select a directory', ".")
		self.lineEdit4.setText(dirName)
		self.outdir = dirName
		print("Output directory selected: ", self.outdir)


	def run_model(self) :
		self.tabs.setCurrentIndex(1)

		print("Running the model...")

		model = self.mcmc_model

		if self.mcmc_model.abort==True :
			model.reset()
		else :
			model.set_up_data()

		model.set_outdir(self.outdir)

		model.set_iterations(self.lineEdit1.text(), self.lineEdit2.text(), self.lineEdit3.text())

		self.externalthread = ExternalThread(model)
		self.externalthread.countChanged.connect(self.onCountChanged)
		self.externalthread.start()

		#model.countChanged.connect(self.onCountChanged)

		# sadly mpl will not work outside the main thread:
		#	"UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail."
		#if model.plotting_switch:
		#	model.run()
		#else :
		#	self.thread = QThread()
		#	self.mcmc_model.moveToThread(self.thread)
		#	self.thread.started.connect(self.mcmc_model.process)
		#	self.thread.start()

		model.run_model()
		self.progressbar.setValue(100)

	def abort_model(self) :
		print("Stop called...")
		self.clear_clicked()
		self.mcmc_model.abort=True
		print("The input data was cleared. Please load it again.")







app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
