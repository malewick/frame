import matplotlib.pyplot as plt
import matplotlib

import pandas as pd

class ResultsPlot:

	def __init__(self,f_out,fmt):
		filename = f_out

		df=pd.read_csv(filename);
		print(df)
	
		variables=[]
		for c in df.columns:
			if "mean" in c:
				variables.append(c[5:])
		print(variables)


		for gr in df["group"].unique():
			dft = df.loc[df["group"]==gr]

			fig, ax = plt.subplots(figsize=(4,3))
			fig.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.98, wspace=0.00)


			for i,s in enumerate(variables):

				ax.errorbar([s],dft["mean_"+s],yerr=[abs(dft["mean_"+s]-dft["CI95_low"+s]),abs(dft["CI95_up"+s]-dft["mean_"+s])],capsize=4, color="C0", zorder=1)
				ax.bar([s], dft["CI68_up"+s]-dft["CI68_low"+s], 0.35, bottom=dft["CI68_low"+s],facecolor="white",edgecolor="C0", zorder=2)
				ax.plot([s],dft["mean_"+s], "o", markeredgecolor="C0", markerfacecolor="none", markersize=8, zorder=3)
				ax.plot([s],dft["median_"+s], "_",color="C0", markersize=18, zorder=4)

				ax.text(i+0.18,dft["CI68_low"+s],"{:.3f}".format(dft["CI68_low"+s].iloc[0]),fontsize=6,va="center")
				ax.text(i+0.18,dft["CI68_up"+s],"{:.3f}".format(dft["CI68_up"+s].iloc[0]),fontsize=6,va="center")
				ax.text(i+0.18,dft["mean_"+s],"{:.3f}".format(dft["mean_"+s].iloc[0]),fontsize=6,va="center")

			ax.set_xlim(-0.3,len(variables)-0.4)
			ax.set_ylim(0,1)
			ax.set_xlabel("variables")
			ax.set_ylabel("fraction")

			for ext in fmt.split(","):
				plt.savefig(f_out[:-4]+"_group_"+str(gr)+"."+ext,dpi=300)
