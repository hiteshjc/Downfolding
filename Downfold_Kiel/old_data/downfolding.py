import matplotlib 
import matplotlib.pyplot as plt
import glob
import json
import sys
#import dm_tools
import numpy as np
#import seaborn as sns
from scipy import stats
#sns.set(style="ticks")
import statsmodels.api as sm
import pandas as pd
plt.rc('text',usetex=True)
plt.rc('lines',linewidth=1)
plt.rc('legend',fontsize=10)
plt.rc('mathtext',fontset='cm')
plt.rc('font',**{'family':'serif','serif':['Helvetica'],'size':10})


#New code to pick-out configurations with the chosen charge configurations
#Also contains code (generate_df_force) to force various parameter to be equal if desired (i.e., U=U', etc.)

dat=json.load(open("cr_data_lucas.json"))
charge_on_int=int(sys.argv[1])
if (charge_on_int==0): charge_on=False
if (charge_on_int==1): charge_on=True
print "charge on =",charge_on
def generate_df(data):
  keys=['atom','energy','error','Nd','Ne','Sz','e','U','Uprime','J','Jprime','Constant','Epsilon','Epsilon_up','Epsilon_down']
  df={}
  for k in keys:
    df[k]=[]
    
  for d in data:
   if 'postprocess' in d['qmc']['dmc'].keys():

    fill=d['filling']
    sz=len(fill[0])-len(fill[1])

    nd=0
    ne=0
    for i in range(6,11):
      if fill[0].count(i) > 0 and fill[1].count(i) > 0:
        nd+=1
      ne+=fill[0].count(i)+fill[1].count(i)
    #print(fill,nd)

    obdm=d['qmc']['dmc']['postprocess'][0]['results']['properties']['tbdm_basis']['obdm']
    tbdm=d['qmc']['dmc']['postprocess'][0]['results']['properties']['tbdm_basis']['tbdm']

    ####Energy

    eterm=0
    for i in range(1,6):
      for s in ['up','down']:
        #print(obdm[s][i][i])
        eterm+=obdm[s][i][i]
   
    # U 
    uterm=0
    for i in range(1,6):
      uterm+=tbdm['updown'][i][i][i][i]
    
    # U' 
    upterm=0
    #This is the first uprime term in Georges' expression
    for i in range(1,6):
        for j in range(1,6):
            if (i!=j):
                upterm+=tbdm['updown'][i][j][i][j]
   #This is the second uprime term in Georges' expression
    for i in range(1,6):
      for j in range(i+1,6):
        for s in ['upup','downdown']:
          upterm+=tbdm[s][i][j][i][j]

    # J
    #jterm=0
    #for i in range(1,6):
    #  for j in range(1,6):
    #      jterm+=0.5*tbdm['upup'][i][j][j][i]
    #      jterm+=0.5*tbdm['downdown'][i][j][j][i]
    #      if (i!=j):    
    #      	jterm+=tbdm['updown'][i][j][j][i]
    
    jterm=0
    for i in range(1,6):
      for j in range(1,6):
          if (i<j): jterm-=tbdm['upup'][i][j][i][j]
          if (i<j): jterm-=tbdm['downdown'][i][j][i][j]
          if (i!=j):    
          	jterm+=tbdm['updown'][i][j][j][i]
            
    # J' - double hopping seems small
    jpterm=0
    for i in range(1,6):
      for j in range(1,6):
          if (i!=j):    
            jpterm+=tbdm['downup'][i][i][j][j]
            
    # 1-RDM trace
    trace_up=0
    trace_down=0
    for i in range(1,6):
      trace_up+=obdm['up'][i][i]#+obdm['down'][i][i]
      trace_down+=obdm['down'][i][i]#+obdm['down'][i][i]
    trace=trace_up+trace_down

    if (d['atom']=='V') or (d['atom']=='Cr' and (ne==3 or ne==4 or ne==5)) or (d['atom']=='Mn'):
      df['J'].append(jterm)
      df['Jprime'].append(jpterm)
      df['Epsilon'].append(trace)
      df['Epsilon_up'].append(trace_up)
      df['Epsilon_down'].append(trace_down)
      df['Uprime'].append(upterm)
      df['U'].append(uterm)
      df['e'].append(eterm)
      df['Constant'].append(1.0)
      df['Nd'].append(nd)
      df['Ne'].append(ne)
      df['atom'].append(d['atom'])
      df['energy'].append(d['qmc']['dmc']['results'][0]['results']['properties']['total_energy']['value'][0])
      df['error'].append(d['qmc']['dmc']['results'][0]['results']['properties']['total_energy']['error'][0])
      df['Sz'].append(sz)

  return pd.DataFrame(df) 

def fit_dataframe(parameters,df):
  nparms=len(parameters)
  npts=len(df)
  A=np.zeros((npts,nparms))
  for i,p in enumerate(parameters):
    A[:,i]=np.array(df[p])
  K=np.array(df['energy'])
  x,res,rank,s=np.linalg.lstsq(A,K)
  K_mod=np.dot(A,x)
#  print(A)
#  print(K)
  Kerr=np.array(df['error'])
  return x,res,K,Kerr,K_mod


def plot_pair(pvals,parms,df,ax,pair,rangex,rangey,fsize=16):
  ev=27.2114
  x=pair[0]
  y=pair[1]
  xn=parms[x]
  yn=parms[y]
  print('X Range')
  print(rangex)
  print('Y Range')
  print(rangey)
  xs=np.linspace(rangex[0],rangex[1],200)
  ys=np.linspace(rangey[0],rangey[1],200)
  X,Y=np.meshgrid(xs,ys)
  xvals=np.einsum('ab,c->abc',X,df[xn])
  yvals=np.einsum('ab,c->abc',Y,df[yn])
  toterr=xvals+yvals-df['energy'][np.newaxis,np.newaxis,:]
  for i,p in enumerate(parms):
    if i!=x and i!=y:
      toterr+=pvals[i]*df[p][np.newaxis,np.newaxis,:]
  err=np.sqrt(np.sum(toterr**2,axis=2)/toterr.shape[2])
  ax.pcolor(ev*X,ev*Y,ev*err,cmap='autumn')
  
  CS=ax.contour(ev*X,ev*Y,ev*err,color='k')  
  ax.clabel(CS,inline=1,fontsize=10,color='k')
  
  ax.set_xlabel(xn+"(eV)",fontsize=fsize)
  ax.set_ylabel(yn+"(eV)",fontsize=fsize)


#Produce energy-energy fit plots, grouping fits together (E0-J Model)

dater_luc=generate_df(dat)
if (charge_on==True):
	groups=dater_luc.groupby(['atom','Ne'])
else:
	groups=dater_luc.groupby(['atom'])
fig,axes2d=plt.subplots(2,int((len(groups)+1)/2),figsize=(10.,6.))
axes=axes2d.flatten()

count=0
pcount=1
evc=27.2114

#f1 = plt.figure(1)
#f2 = plt.figure(2)
#f3 = plt.figure(3)

for name, group in groups:
#  if group['atom'].any()=='V' or group['atom'].any()=='Mn':
#    print('yes')
  #print "Name"    
  print(name)
  #  print(group)
  #  x=fit_dataframe(['E0','U','Uprime','J'],group()
  #paramers=['Constant','U','J']
  #paramers=['Constant','U','Uprime','Epsilon','J']
  paramers=['Constant','Epsilon','U','Uprime','J']
  x=fit_dataframe(paramers,group)

  #  print(x)
  #  print('x')
  #  print(x)
  #  print(x,27.2114*res)
  print(group.sort('Sz').head(100).to_string())
  #print(group.head(100).to_string())

    
  cc=0
  for k in paramers:
    print(k)
    print(x[0][cc]*evc)
    cc+=1

  natom = len(group['Constant'])
#  print('Residual (eV)')
#  print(np.sqrt(x[1][0]/natom)*evc)

  E_ab=x[2]
  E_err=x[3]
  E_mod=x[4]

#  plt.figure(pcount)
  set=0.1
  print('Ab-Initio Energies')
  print(E_ab)
  print('Model Energies')
  print(E_mod)

  axes[count].errorbar(E_mod, E_ab, yerr=E_err, linestyle="",lw=1,marker="o",markersize=3,mew=2)
  axes[count].plot([np.min(E_ab)-set,np.max(E_ab)+set],[np.min(E_ab)-set,np.max(E_ab)+set],lw=2)

  #plt.plot([-100,-2],[-100,-2],lw=2)
  axes[count].set_xlim([np.min(E_mod)-set, np.max(E_mod)+set])
  axes[count].set_ylim([np.min(E_ab)-set, np.max(E_ab)+set])

#  plt.gcf().tight_layout()
#  plt.savefig('atomfit'+str(pcount)+'.jpg')
#  pcount+=1
  fsize=10
  axes[count].set_xlabel("Model Energy (Hartree)",fontsize=fsize)
  axes[count].set_ylabel("Ab-Initio Energy (Hartree)",fontsize=fsize)
  if (charge_on):
	axes[count].set_title(name[0]+str(name[1]))
  else:
	axes[count].set_title(name[0])
  plt.setp(axes[count].get_xticklabels(), rotation=40, horizontalalignment='right')
  count+=1
    

  #print(group)
  #print(group.corr())
#  if len(group)>=3:
#    print(fit_dataframe(['e','U'],group))
#  if len(group)>=5:
#    print(fit_dataframe(['e','U','Uprime'],group))
#    print(fit_dataframe(['e','U','J','Uprime'],group))
#  print(group)

plt.tick_params(axis='both', which='major', labelsize=3)
plt.tick_params(axis='both', which='minor', labelsize=3)
plt.tight_layout()
#plt.savefig("Fit_Out.jpg")
plt.show()

