import pylab 

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
		pylab.plot(nl[1],nl[4],color="black",marker="o",markersize=10)
		pylab.plot(nl[1],nl[5],color="red",marker="^",markersize=10)
		pylab.plot(nl[1],nl[6],color="black",marker="o",markersize=10)
		pylab.plot(nl[1],nl[7],color="red",marker="^",markersize=10)
		pylab.plot(nl[1],nl[8],color="black",marker="o",markersize=10)
		pylab.plot(nl[1],nl[9],color="red",marker="^",markersize=10)
		pylab.xlim([3.5,12.5])
	if (nl[0]==5.0):
		pylab.figure(2)
		pylab.plot(nl[1],nl[4],color="black",marker="o",markersize=10)
		pylab.plot(nl[1],nl[5],color="red",marker="^",markersize=10)
		pylab.plot(nl[1],nl[6],color="black",marker="o",markersize=10)
		pylab.plot(nl[1],nl[7],color="red",marker="^",markersize=10)
		pylab.plot(nl[1],nl[8],color="black",marker="o",markersize=10)
		pylab.plot(nl[1],nl[9],color="red",marker="^",markersize=10)
		pylab.xlim([3.5,12.5])
pylab.show()
