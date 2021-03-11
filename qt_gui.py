import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PySide2.QtCore import Qt
from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


import pandas as pd
import numpy as np


from multiprocessing import Process, Queue


import NDimModel	


class EmittingStream(QtCore.QObject):

	textWritten = QtCore.Signal(str)

	def write(self, text):
		self.textWritten.emit(str(text))



class MplCanvas(FigureCanvasQTAgg):


	def __init__(self, parent=None, width=5, height=4, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
		super(MplCanvas, self).__init__(self.fig)

	def plot_data(self, df, arg1, arg2):
		for i, group in enumerate(df['group'].unique()):
			dfs=df.loc[df['group']==group]
			print(dfs[arg1].tolist(), dfs[arg2].tolist())
			self.axes.scatter(dfs[arg1].tolist(), dfs[arg2].tolist(),
							  edgecolor="C"+str(i%10),
							  #facecolor="C"+str(i%10),
							  facecolor="None",
							  marker="$"+str(group)+"$",
							  #marker='o',
							  alpha=1.,
			)
			self.axes.set_xlabel(arg1)
			self.axes.set_ylabel(arg2)

			
			self.figure.canvas.draw()

	def plot_sources(self, df_model, arg1, arg2):
		# plotting the sources with uncertainties as rectangles 
		for j, tstr in enumerate(df_model["source"].unique()) :
				vx=df_model.loc[(df_model['source'] == tstr) & (df_model['isotope'] == arg1)].iloc[0]['value']
				sx=df_model.loc[(df_model['source'] == tstr) & (df_model['isotope'] == arg1)].iloc[0]['stdev']
				vy=df_model.loc[(df_model['source'] == tstr) & (df_model['isotope'] == arg2)].iloc[0]['value']
				sy=df_model.loc[(df_model['source'] == tstr) & (df_model['isotope'] == arg2)].iloc[0]['stdev']
				print(vx,vy,sx,sy)
				rect = Rectangle((vx-sx,vy-sy),2*sx,2*sy,linewidth=1,edgecolor='black',facecolor="black",alpha=0.15)
				self.axes.add_patch(rect)
				self.axes.text(vx-sx+0.5,vy-sy+0.5, tstr, fontsize=12)

		# plotting polygons bounded by the sources
		#xs=[]
		#ys=[]
		#for tstr in df_model["source"].unique() :
		#	print("-----")
		#	print(tstr, arg1, arg2)
		#	print((df_model.loc[(df_model['source'] == tstr)]))
		#	print((df_model.loc[(df_model['isotope'] == arg1)]))
		#	print((df_model.loc[(df_model['isotope'] == arg2)]))
		#	xs.append(df_model.loc[(df_model['source'] == tstr) & (df_model['isotope'] == arg1)].iloc[0]['value'])
		#	ys.append(df_model.loc[(df_model['source'] == tstr) & (df_model['isotope'] == arg2)].iloc[0]['value'])
		##print(xs)
		##print(ys)

		#polygon = Polygon(np.array(list(zip(xs,ys))), True, edgecolor='black',facecolor='none',alpha=0.1)
		#self.axes.add_patch(polygon)
		self.axes.autoscale()
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



class MainWindow(QtWidgets.QMainWindow):

 

	def __init__(self, *args, **kwargs):
		super(MainWindow, self).__init__(*args, **kwargs)

		self.df_data = pd.DataFrame(data={'-':[0]})
		self.df_srcs = pd.DataFrame(data={'-':[0]})
		self.df_auxs = pd.DataFrame(data={'-':[0]})

		self.mcmc_model = NDimModel.Model()

		self.table1 = QtWidgets.QTableView()
		self.model1 = TableModel(self.df_data)
		self.table1.setModel(self.model1)

		self.table2 = QtWidgets.QTableView()
		self.model2 = TableModel(self.df_srcs)
		self.table2.setModel(self.model2)

		self.table3 = QtWidgets.QTableView()
		self.model3 = TableModel(self.df_auxs)
		self.table3.setModel(self.model3)

		self.outdir = ""

		self.sc1 = MplCanvas(self, width=5, height=4, dpi=100)

		self.toolbar1 = NavigationToolbar(self.sc1, self)

		self.sc2 = MplCanvas(self, width=5, height=4, dpi=100)

		self.toolbar2 = NavigationToolbar(self.sc2, self)

		self.sc3 = MplCanvas(self, width=5, height=4, dpi=100)

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

		button_load = QtWidgets.QPushButton("1. Load samples")
		button_sources = QtWidgets.QPushButton("2. Load sources")
		button_frac = QtWidgets.QPushButton("3. Load aux. data \n (optional)")
		button_save = QtWidgets.QPushButton("Save plots")

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

		button_load.clicked.connect(self.load_data)
		button_sources.clicked.connect(self.load_sources)
		button_frac.clicked.connect(self.load_aux)


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

		layout_aux = QtWidgets.QVBoxLayout()
		label_model = QtWidgets.QLabel("Model equation:")
		label_model.setFont(QtGui.QFont('Sans', 11)) 
		label_derivatives = QtWidgets.QLabel("Model derivatives:")
		label_derivatives.setFont(QtGui.QFont('Sans', 11)) 
		self.ledit_model = QtWidgets.QLineEdit("");
		self.ledit_derivatives = QtWidgets.QLineEdit("");
		layout_aux.addWidget(label_model)
		layout_aux.addWidget(self.ledit_model)
		layout_aux.addWidget(label_derivatives)
		layout_aux.addWidget(self.ledit_derivatives)

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
		layout_allothers.addLayout(layout_aux)

		layout_data = QtWidgets.QGridLayout()
		layout_data.addLayout(layout_buttons1,0,0,1,1)
		layout_data.addLayout(layout_buttons2,0,1,1,1)
		layout_data.addLayout(layout_buttons3,0,2,1,1)
		layout_data.addWidget(self.table1,1,0)
		layout_data.addWidget(self.table2,1,1)
		layout_data.addWidget(self.table3,1,2)
		layout_data.setRowStretch(0,1)
		layout_data.setRowStretch(1,3)
		layout_data.addLayout(layout_allothers,0,3,2,1)


		layout_data2 = QtWidgets.QGridLayout()
		layout_data2.addLayout(layout_data,0,0,1,3)
		layout_data2.addLayout(layout1,1,0)
		layout_data2.addLayout(layout2,1,1)
		layout_data2.addLayout(layout3,1,2)


	# -----
	# ----- Second card
	# -----

		label1 = QtWidgets.QLabel("Number of iterations: ");
		label2 = QtWidgets.QLabel("Burn-out iterations: ");
		label3 = QtWidgets.QLabel("Desired Markov-chain length: ");
		
		self.lineEdit1 = QtWidgets.QLineEdit("1000000");
		self.lineEdit2 = QtWidgets.QLineEdit("10000");
		self.lineEdit3 = QtWidgets.QLineEdit("500");
		
		formLayout = QtWidgets.QFormLayout();
		formLayout.addRow(label1, self.lineEdit1)
		formLayout.addRow(label2, self.lineEdit2)
		formLayout.addRow(label3, self.lineEdit3)

		#button2 = QtWidgets.QPushButton("Abort")

		label4 = QtWidgets.QLabel("Output directory: ");
		self.lineEdit4 = QtWidgets.QLineEdit("./output/");
		button4 = QtWidgets.QPushButton("Set...")
		button4.clicked.connect(self.set_outdir)
		formLayout2 = QtWidgets.QFormLayout();
		formLayout2.addRow(label4, self.lineEdit4)
		layout_outdir = QtWidgets.QHBoxLayout()
		layout_outdir.addLayout(formLayout2)
		layout_outdir.addWidget(button4)
		layout_outdir.addStretch(1)

		button1 = QtWidgets.QPushButton("4. Run model")
		button1.setFont(QtGui.QFont('Sans', 15)) 
		button1.clicked.connect(self.run_model)
	
		layout_buttons = QtWidgets.QHBoxLayout()
		layout_buttons.addWidget(button1)
		#layout_buttons.addWidget(button2)
		layout_buttons.addStretch(1)

		layout_h = QtWidgets.QVBoxLayout()
		layout_h.addLayout(formLayout)
		layout_h.addLayout(layout_outdir)
		layout_h.addLayout(layout_buttons)
		#layout_h.addStretch(1)

		self.logConsole = QtWidgets.QTextEdit()

		layout_model = QtWidgets.QVBoxLayout()
		layout_model.addLayout(layout_h)
		layout_model.addWidget(self.logConsole)

		tabs = QtWidgets.QTabWidget()

		widget_data = QtWidgets.QWidget()
		widget_data.setLayout(layout_data2)
		tabs.addTab(widget_data, "Input Data & Model")

		widget_model = QtWidgets.QWidget()
		widget_model.setLayout(layout_model)
		tabs.addTab(widget_model, "Running MCMC")

		# Create a placeholder widget to hold our toolbar and canvas.
		#widget = QtWidgets.QWidget()
		#widget.setLayout(layout)
		#self.setCentralWidget(widget)
		self.setCentralWidget(tabs)

		menuBar = self.menuBar()
		fileMenu = menuBar.addMenu("&File");
		viewMenu = menuBar.addMenu("&View");

		action_new = QtWidgets.QAction("&New", self);
		action_new.setStatusTip("Start a new analysis from scratch")
		#action_new.triggered.connect(self.show_hide_toolbar)
		fileMenu.addAction(action_new);

		action_save = QtWidgets.QAction("&Save", self);
		action_save.setStatusTip("Save session")
		#action_save.triggered.connect(some_function)
		fileMenu.addAction(action_save);

		action_load = QtWidgets.QAction("&Load", self);
		action_load.setStatusTip("Load session")
		#action_load.triggered.connect(some_function)
		fileMenu.addAction(action_load);


		self.action_mpl_toolbar = QtWidgets.QAction("&Plotting toolbar", self);
		self.action_mpl_toolbar.setCheckable(True);
		self.action_mpl_toolbar.setChecked(False);
		self.action_mpl_toolbar.setStatusTip("Show toolbars for plotting")
		self.action_mpl_toolbar.triggered.connect(self.show_hide_toolbar)
		viewMenu.addAction(self.action_mpl_toolbar);



		es = EmittingStream()
		sys.stdout = es
		self.connect(sys.stdout,QtCore.SIGNAL('textWritten(QString)'),self.normalOutputWritten)
		#sys.stdout = EmittingStream()
		#self.connect(sys.stdout,QtCore.SIGNAL('textWritten(QString)'),self.write2Console)




		self.show()


	def __del__(self):
		# Restore sys.stdout
		sys.stdout = sys.__stdout__

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

	def load_data(self) :
		print("Loading data on measurements...")
		fileName = QtWidgets.QFileDialog.getOpenFileName(self,
				"Open File", "./test_input/", "CSV Files *.csv")
		print(fileName[0])

		self.mcmc_model.load_measurements(fileName[0])

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

		

	def load_sources(self) :
		print("Loading data on sources...")
		fileName = QtWidgets.QFileDialog.getOpenFileName(self,
				"Open File", "./test_input/", "CSV Files *.csv")

		self.mcmc_model.load_sources(fileName[0])

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


	def load_aux(self) :
		print("Loading aux data...")
		fileName = QtWidgets.QFileDialog.getOpenFileName(self,
				"Open File", "./test_input/", "CSV Files *.csv")

		self.mcmc_model.load_aux(fileName[0])

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


	def set_outdir(self) :
		print("Choosing output directory...")
		dirName = QtWidgets.QFileDialog.getExistingDirectory( self,
				'Select a directory', ".")
		self.lineEdit4.setText(dirName)
		self.outdir = dirName
		print("Output directory selected: ", self.outdir)


	def run_model(self) :
		print("Running the model...")

		model = self.mcmc_model

		model.set_up_data()

		model.set_outdir(self.outdir)

		model.set_iterations(self.lineEdit1.text(), self.lineEdit2.text(), self.lineEdit3.text())

		model.run_model()







app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
