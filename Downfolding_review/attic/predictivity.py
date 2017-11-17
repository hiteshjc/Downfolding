import pylab 

ctr=0
f=open("plot_predictivity.txt",'r')
for line in f:
	line=line.strip()
	line=line.split(" ")
	nl=[]
	for a in line:
		if (a!=""): 
			nl.append(float(a))
	print nl
	if (nl[0]==3.0):
		pylab.figure(1)
		pylab.plot(nl[1],nl[4],color="black",marker="o",markersize=17,markeredgecolor="black")
		pylab.plot(nl[1],nl[5],color="red",marker="^",markersize=12,markeredgecolor="red")
		pylab.plot(nl[1],nl[6],color="black",marker="o",markersize=17,markeredgecolor="black")
		pylab.plot(nl[1],nl[7],color="red",marker="^",markersize=12,markeredgecolor="red")
		pylab.plot(nl[1],nl[8],color="black",marker="o",markersize=17,markeredgecolor="black")
		pylab.plot(nl[1],nl[9],color="red",marker="^",markersize=12,markeredgecolor="red")
		pylab.xlim([3.5,12.5])
	if (nl[0]==5.0):
		pylab.figure(2)
		if (ctr==0):
			pylab.plot(nl[1],nl[4],color="black",marker="o",markersize=17,markeredgecolor="black",label="3-band gap")
			pylab.plot(nl[1],nl[5],color="red",marker="^",markersize=12,markeredgecolor="red",label="1-band gap")
			ctr=1
		else:
			pylab.plot(nl[1],nl[4],color="black",marker="o",markersize=17,markeredgecolor="black")
			pylab.plot(nl[1],nl[5],color="red",marker="^",markersize=12,markeredgecolor="red")
		pylab.plot(nl[1],nl[6],color="black",marker="o",markersize=17,markeredgecolor="black")
		pylab.plot(nl[1],nl[7],color="red",marker="^",markersize=12,markeredgecolor="red")
		pylab.plot(nl[1],nl[8],color="black",marker="o",markersize=17,markeredgecolor="black")
		pylab.plot(nl[1],nl[9],color="red",marker="^",markersize=12,markeredgecolor="red")
		pylab.xlim([3.5,12.5])
pylab.figure(1)
pylab.xlabel("$U_d/t_{pd}$",fontsize=40)
pylab.ylabel("Energy gap (eV)",fontsize=40)
pylab.xticks(fontsize=33)
pylab.yticks(fontsize=33)
pylab.text(8,0.075,"$\Delta/t_{pd} = 3$",fontsize=50)
pylab.savefig("../Figures/Predict_8site_Ep_3.eps",bbox_inches="tight")

pylab.figure(2)
pylab.xlabel("$U_d/t_{pd}$",fontsize=40)
pylab.ylabel("Energy gap (eV)",fontsize=40)
pylab.text(8,0.06,"$\Delta/t_{pd} = 5$",fontsize=50)
pylab.xticks(fontsize=33)
pylab.yticks(fontsize=33)
pylab.legend(numpoints=1,loc="best",prop={'size': 25})
pylab.savefig("../Figures/Predict_8site_Ep_5.eps",bbox_inches="tight")
pylab.show()
