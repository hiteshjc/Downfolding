''' Analyze data from FeSe diatomic molecule. '''
import sys
#sys.path.append('/home/brian/programs/downfolding')

import os
import pickle as pkl
import downfold_tools as dt
import time
import numpy as np
import statsmodels.api as sm
from pyscf import lib,scf,lo,gto
import pandas as pd
import matplotlib.pyplot as plt
import busempyer.plot_tools as pt # part of github.com/bbusemeyer/busempyer
import seaborn as sns
from copy import copy, deepcopy
#import gather_data as gather
ev=27.2114
pt.matplotlib_header()

fese='\n'.join([
    'Fe 0. 0. 0.',
    'Se 0. 0. 2.43'
  ])

minbasis={
    'Fe':gto.basis.parse('''
      Fe s
      25.882657     0.000341
      14.037755     -0.047646
      9.007794     0.136465
      2.068350     -0.153371
      0.993498     -0.288555
      0.471151     -0.044212
      0.102489     0.704768
      0.036902     0.415407
      Fe d
      10.232413     0.081591
      4.841151     0.263190
      2.039827     0.342856
      0.840565     0.338326
      0.328485     0.240730
      0.116926     0.088375
      '''),
    'Se':gto.basis.parse('''
      Se p
      0.056147     0.073504
      0.122259     0.334692
      0.266220     0.473323
      0.579694     0.276571
      1.262286     -0.032356
      2.748631     -0.103709
      5.985152     0.020181
      13.032685     -0.001095
      28.378708     0.000019
      ''')
  }


# List of all possible descriptors names.
desc=[
    'es',
    'eddel', # dxy,dx2-y2
    'edz2', # dz2
    'edpi', # dxz,dyz
    'epz',
    'eppi',
    'tsigs',
    'tsigd',
    'tfe',
    'tpi',
    'Ud',
    'Ud0',
    'Udel',
    'Upi',
    'Uz2',
    'Uz2,pi',
    'Uz2,del',
    'Upi,del',
    'Up',
    'J',
    'Jpi', 
    'Jdel', 
    'Jz2,pi',
    'Jz2,del',
    'Jpi,del',
    'V'
  ]

simple_parms=['E0','J','tsigd','Ud','tsigs','edxy','es','epz']
best_parms=['E0','J','tsigd','Ud','tsigs','edxy','es','epz','tpi','Jxzyz','Jxy','Uz2','tfe']

parmmap={
    'E0'         :  r'$E_0$',
    'RMSE'       :  'RMSE',
    'es'         :  r'$\epsilon_s$',
    'edxy'       :  r'$\epsilon_{\delta,\mathrm{Fe}}$',
    'edz2'       :  r'$\epsilon_{d_{z^2}}$',
    'edxzyz'     :  r'$\epsilon_{d_{xz,yz}}$',
    'epz'        :  r'$\epsilon_{p_z}$',
    'epxy'       :  r'$\epsilon_{p_{xy}}$',
    'tsigs'      :  r'$t_{\sigma,s}$',
    'tsigd'      :  r'$t_{\sigma,d}$',
    'tpi'        :  r'$t_\pi$',
    'tfe'        :  r'$t_\mathrm{Fe}$',
    'Ud'         :  r'$U_d$',
    'Up'         :  r'$U_p$',
    'Uxy'        :  r'$U_{xy}$', 
    'Uxzyz'      :  r'$U_{xzyz}$',
    'Uz2'        :  r'$U_{z^2}$',
    'J'          :  r'$J$',
    'Jxy'        :  r'$J_{xy,xy}$',
    'Jxy,xzyz'   :  r'$J_{xy,xzyz}$',
    'Jxy,z2'     :  r'$J_{xy,z^2}$',
    'Jxzyz'      :  r'$J_{xzyz,xzyz}$',
    'Jxzyz,z2'   :  r'$J_{xzyz,z^2}$',
    'V'          :  r'$V$'
  }

def _print_mat(mat,labs,args={},div=-1.0):
  ''' Useful for checking the symmetries of the DM. '''
  fig,axes=plt.subplots(1,1)
  axes.axhline(div)
  axes.axvline(div)
  im=axes.matshow(mat,**args)
  fig.colorbar(im)
  axes.set_xticks(range(len(labs)))
  axes.set_yticks(range(len(labs)))
  axes.set_xticklabels(labs,rotation=90)
  axes.set_yticklabels(labs)
  fig.set_size_inches(4,4)
  fig.tight_layout()
  return fig,axes

def fit_FeSe():
  ''' Perform the fits and export to models and data to fitdf.pkl.'''
  
  scfdf=pd.read_csv('scfdf.csv')
  qmcdf=pd.read_csv('qmcdf.csv')
  fitdf=qmcdf # hacky way of choosing dataset.
  fit_procedure(fitdf,parms=['J','tsigd','Ud'])

def fit_procedure(fitdf,parms=copy(best_parms)):
  ''' Fit models to fit df and save results to fitdf.pkl.

  Args:
    parm (list): MP order to use.
    fitdf (DataFrame): Dataframe of energies, errors, and descriptors in desc.
  Returns:
    (list): models of parameters added according to parm.
  '''

  print("Size before:",fitdf.shape)

  # Some filtering of the database.
  outofspace=abs(fitdf['totocc']-12)>0.5
  print("Out of space: %d"%(outofspace.sum()))
  #print(fitdf[outofspace])
  #print(fitdf.shape)
  fitdf=fitdf[~outofspace].copy().reset_index(drop=True)

  ewin=8/ev
  high=fitdf['energy']-fitdf['energy'].min()>ewin
  print("High energy: %d"%(high.sum()))
  fitdf=fitdf[~high].copy().reset_index(drop=True)

  print("Size after:",fitdf.shape)

  ## Parallel axis plot.
  #cols=['energy','totocc']+desc
  #parallel(fitdf[cols].copy(),
  #    ref=fitdf.loc[fitdf['energy']==fitdf['energy'].min(),cols].squeeze())

  # Visualize correlation matrix.
  cols=['energy']+desc
  fig,axes=_print_mat(fitdf[cols].corr(),labs=cols,
      args={'cmap':'BrBG','vmin':-1.0,'vmax':1.0})
  fig.savefig('corr.pdf')
  fig.savefig('corr.eps')
  print("Saved corr.pdf/eps")

  ## PCA on the descriptor space.
  #eigval,eigvec=mypca(fitdf[desc])
  #fig,axes=_print_mat(eigvec.T,labs=desc,
  #    args={'cmap':'BrBG','vmin':-abs(eigvec.max()),'vmax':abs(eigvec.max())})
  #axes.set_xticklabels(eigval.round(2))
  #fig.savefig('desc_pca.pdf')
  #fig.savefig('desc_pca.eps')
  #print("Saved desc_pca.pdf/eps")

  # Fit steps.
  fitdf['E0']=1.0
  models=fit_steps(
  #models=quick_fit(
      fitdf[['id','extid','excitation','energy','energy_err','E0']+desc],
      parms,desc+['E0'],
      prefix='')
  with open('fitdf.pkl','wb') as outf:
    pkl.dump((fitdf[['id','extid','excitation','energy','energy_err','E0']+desc],models),outf)

  #print("Model:")
  #print(pd.DataFrame({'parm':models[-1].params,'error':models[-1].bse})*ev)

  #print("RMSE:")
  #print([(m.resid**2).mean()**0.5 * ev for m in models])

  #print("BIC:")
  #print([m.bic * ev for m in models])

  return models

def fit_model(df,parcols,fitcol,errcol):
  """ Wrapper for OLS."""
  model = sm.OLS(df[fitcol],df[parcols],hasconst=True)
  result = model.fit()
  return result

def analyze_fit(fig,axes,df,allparms,fitparms):
  """ Fit model and report errors, plot residuals."""
  model = fit_model(
      df,
      fitparms,
      'energy',
      'energy_err'
      )

  leftparms = [parm for parm in allparms if parm not in fitparms]
  cors=[]
  #fig,axes = plt.subplots(1,len(leftparms),sharey=True,squeeze=False)
  for pidx,parm in enumerate(leftparms):
    ax = axes[pidx]
    cor = np.corrcoef(np.array([df[parm],model.resid]))[0,1]
    cors.append(cor)
    ax.set_title("Cor:%.2f"%cor)
    sel=df['id'].apply(lambda x: 'rks' in x)
    ax.plot(df.loc[sel,parm],model.resid[sel]*ev,'.',color=pt.pc['b'],alpha=0.9)
    sel=df['id'].apply(lambda x: 'rks' not in x)
    ax.plot(df.loc[sel,parm],model.resid[sel]*ev,'.',color=pt.pc['b'],alpha=0.9)
    ax.set_xlabel("$\Sigma(\mathrm{%s})$"%(parm))
  #axes[0].annotate("Total MSE: %.2e"%((model.resid**2).sum()*ev),(0.5,0.95),xycoords='axes fraction',ha='center')
  if len(axes)>0: # Sometimes there's no parameters left.
    axes[0].set_ylabel("Residual (eV)")
  sns.despine()

  cordf=pd.DataFrame({'parm':leftparms,'cor':cors})
  cordf['abscor']=cordf['cor'].abs()
  print("Correlation coefficients")
  print(cordf.sort_values('abscor',ascending=False))
  return fig,axes,model

def quick_fit(df,parmpath,allparms,prefix=None):
  """Fit a sequence of models adding one parameter at a time. Skip plotting."""
  fitparms = []
  models = []
  for parm in parmpath:
    fitparms.append(parm)
    models.append(fit_model(df,fitparms,'energy','energy_err'))

  return models

def fit_steps(df,parmpath,allparms,prefix=''):
  """Fit a sequence of models adding one parameter at a time."""
  fitparms = []
  models = []
  fig,axes = plt.subplots(len(parmpath),len(allparms)-1,squeeze=False)

  # Plot residuals.
  print("Performing fits")
  axmap = dict(zip([p for p in allparms if p!=parmpath[0]],range(len(allparms)-1)))
  for fidx,fitparm in enumerate(parmpath):
    fitparms.append(fitparm)
    use_axes = [axmap[parm] for parm in allparms if parm not in fitparms]
    _,_,model = analyze_fit(fig,axes[fidx,use_axes],df,allparms,fitparms)
    for aidx,ax in enumerate(axes[fidx]):
      if aidx not in use_axes:
        plt.delaxes(ax)
    models.append(model)
  for ax in axes.flatten():
    #pt.fix_lims(ax)
    ax.set_xticks(pt.thin_ticks(ax.get_xticks()))
  fig.set_size_inches(2*axes.shape[1],2*axes.shape[0])
  fig.tight_layout()
  for ftype in ['.pdf','.eps']:
    fig.savefig(prefix+'parm_path'+ftype)
  print("Saved %s.parm_path.pdf/eps"%prefix)

  ## Extra informaion. 
  ## Print worst offenders.
  #print("Worst offenders")
  #resid=df['energy'] - model.fittedvalues
  #worst=np.argsort(-resid)[:50]
  #print(pd.DataFrame({'id':df['id'][worst],'extid':df['extid'][worst],
  #  'energy':df['energy'][worst]*ev,'model':model.fittedvalues[worst]*ev,'resid':resid[worst]*ev}))

  ## Weird state.
  #state='1_6_9'
  #print("Investigate weird state %s"%state)
  #sel=(df['id']=='spin_4')&(df['extid']==state)
  #gnd=df['energy']==df['energy'].min()
  #print("Model difference",(model.fittedvalues[sel].reset_index()-model.fittedvalues[gnd].reset_index())*ev)
  #print("Itemizing...")
  #diff=(df.loc[sel,allparms].reset_index(drop=True)-df.loc[gnd,allparms].reset_index(drop=True)).squeeze()
  #diffdf=pd.DataFrame({'params':model.params,'perr':model.bse,'desc_diff':diff})
  #diffdf.loc[diffdf['params'].isnull(),['params','perr']]=0.0
  #diffdf['energy_contibution']=diffdf['desc_diff']*diffdf['params']
  #diffdf[['params','perr','energy_contibution']]*=ev
  #print(diffdf)
  #print(diffdf['energy_contibution'].sum())

  print("Finished parm_path.")
  return models

def plot_mses(fitdf,models,figname="mse_path",axes=None):
  parmpath=models[-1].params.index
  mses=[(model.resid**2).mean()**0.5 for model in models]

  # Plot MSEs
  print("Plotting RMS")
  fmse,axes = plt.subplots(1,1)
  axes.axhline(0.0,color='k',lw=1)
  #axes.plot(range(len(parmpath)-1),mses[1:],'-',color=pt.pc['b'])
  poss = np.arange(len(parmpath))-0.4
  axes.bar(poss,mses)#,'o',color=pt.pc['b'])
  axes.set_xlabel('Parmeters added')
  axes.set_ylabel('RMS Error (eV)')
  axes.set_xticks(poss+0.4)
  axes.set_xticklabels([parmmap[p] for p in parmpath],rotation=90,va='top',ha='center')
  #pt.fix_lims(axes,factor=0.2)
  #axes.set_ylim([-0.1,axes.get_ylim()[1]])
  fmse.set_size_inches(3,3)
  fmse.tight_layout()
  for ftype in ['.pdf','.eps']:
    fmse.savefig(figname+ftype)
  print("Saved %s.pdf/eps"%figname)

def plot_params(fitdf,models,figname="parm_vals"):
  moddf=pd.DataFrame([m.params for m in models]).reset_index()
  errdf=pd.DataFrame([m.bse for m in models]).reset_index()
  moddf['RMSE']=[(m.resid**2).mean()**0.5 for m in models]
  #paramsparamsprint(moddf*ev)
  moddf=moddf.melt(id_vars='index',var_name='parm')
  errdf=errdf.melt(id_vars='index',var_name='parm',value_name='error')
  moddf.loc[moddf['value'].isnull(),'value']=0.0
  errdf.loc[errdf['error'].isnull(),'error']=0.0
  moddf=pd.merge(moddf,errdf,on=['index','parm'],how='outer')
  moddf[['value','error']]*=ev

  first=moddf[(moddf['parm']!='RMSE')&(abs(moddf['value'])>1e-10)]\
      .groupby('parm').agg({'index':min}).reset_index()
  first=first.merge(moddf[['parm','index','value']],on=['parm','index'])
  #print(first)

  fig,axes=plt.subplots(1,2,sharex=True,sharey=False)
  #axes[0].annotate('(a)',xy=(0.1,0.9),xycoords='axes fraction')
  #axes[1].annotate('(b)',xy=(0.9,0.9),xycoords='axes fraction')

  sel=(moddf['parm']!='E0')&(moddf['parm']!='RMSE')
  cp=pt.CategoryPlot(moddf[sel].copy(),
      color='parm',mark='parm')
  axes[0].axhline(0,color='k',lw=1)
  cp.subplot(axes[0],'index','value','error',plotargs=pt.make_plotargs(),line=True)
  print(first.sort_values('index')['parm'].values)
  parmlabels=[parmmap[p] for p in first.sort_values('index')['parm'].values]
  for ax in axes:
    #ax.set_xticks(range(len(parmlabels)))
    #ax.set_xticklabels(parmlabels)
    ax.set_xlabel('MP step')
  axes[0].set_ylabel('Parameter (eV)')
  axes[1].set_ylabel('RMS Error (eV)')
  pt.fix_lims(axes[0])

  axes[1].bar(moddf.loc[moddf['parm']=='RMSE','index']-0.4,moddf.loc[moddf['parm']=='RMSE','value'])
  #cp.add_legend(args={'bbox_to_anchor':(1.0,1.0),'loc':'upper left'},ax=axes[0],labmap=parmmap)
  pshift={
        'tsigd':(-0.7,0.0),
        'Ud':(0.0,0.3),
        'J':(-0.5,-0.1),
        'tsigs':(0.5,0.3),
        'es':(-0.5,0.0),
        'edxy':(0.0,-0.3),
        'epz':(-0.4,-0.2),
        'tpi':(-0.2,-0.25),
        'edz2':(0.8,-0.2)
      }
  for i,info in first.iterrows():
    if info['parm']=='E0': continue
    sh=pshift[info['parm']]
    axes[0].annotate(
        parmmap[info['parm']],
        xy=(info['index'],info['value']),
        xytext=(info['index']+sh[0],info['value']+sh[1]),
        ha='center',va='center')

  sns.despine(fig)
  fig.set_size_inches(6,3)
  #fig.set_size_inches(4,3)
  fig.tight_layout()
  #fig.subplots_adjust(wspace=0.75)
  #fig.subplots_adjust(right=0.7)

  fig.savefig('%s.pdf'%figname)
  fig.savefig('%s.eps'%figname)
  print("Saved %s.pdf/eps"%figname)

def plot_bic(fitdf,models,figname="measure_path"):
  fig,axes=plt.subplots(4,1,sharex=True)

  axes[0].plot(range(len(models)),[(m.resid**2).mean()**0.5*ev for m in models])
  axes[0].set_xlabel("MP step")
  axes[0].set_ylabel("RMSE")

  axes[1].plot(range(len(models)),[m.bic for m in models])
  axes[1].set_xlabel("MP step")
  axes[1].set_ylabel("BIC")

  axes[2].plot(range(len(models)),[m.rsquared for m in models])
  axes[2].set_xlabel("MP step")
  axes[2].set_ylabel("$R^2$")
  print('R2',[m.rsquared for m in models])
  print('RMSE',[(m.resid**2).mean()**0.5*ev for m in models])

  axes[3].plot(range(len(models)),[m.rsquared_adj for m in models])
  axes[3].set_xlabel("MP step")
  axes[3].set_ylabel("a-$R^2$")

  for ax in axes:
    pt.fix_lims(ax)

  fig.set_size_inches(4,0.5+2.5*axes.shape[0])
  fig.tight_layout()
  fig.savefig(figname+".pdf")
  fig.savefig(figname+".eps")
  print("Saved %s.pdf/eps"%figname)

def plot_corr(fitdf,models,figname="corr_path"):
  view=['es','edxy','edz2','epz', 'tsigs','tsigd','tpi', 'Ud0','J']
  models=models[:-1]

  cor={}
  for parm in view:
    if parm=='E0':continue
    cor[parm] = [
        abs(np.corrcoef(np.array([fitdf[parm],model.resid]))[0,1])
        for model in models]
  cordf=pd.DataFrame(cor).reset_index()
  #print(cordf)
  cordf=cordf.melt(id_vars='index',var_name='parm')
  cordf.loc[cordf['value'].isnull(),'value']=0.0
  #cordf['value']*=ev
  cp=pt.CategoryPlot(cordf,color='parm',mark='parm')
  #cp.axes[0,0].annotate('(c)',xy=(0.9,0.9),xycoords='axes fraction')
  cp.axes[0,0].axhline(0,color='k',lw=1)
  cp.plot('index','value',plotargs=pt.make_plotargs(),line=True)
  cp.axes[0,0].set_xlabel('MP step, exclude $J$')
  cp.axes[0,0].set_ylabel('Correlation with Residual')
  pt.fix_lims(cp.axes)
  pmap=copy(parmmap)
  pmap['E0']='$E_0$/1000'
  #cp.add_legend(args={'bbox_to_anchor':(1.0,1.0),'loc':'upper left'},labmap=pmap)
  sns.despine()
  cp.fig.set_size_inches(3.5,3)
  cp.fig.tight_layout()
  #cp.fig.subplots_adjust(right=0.7)
  cp.fig.savefig("%s.pdf"%figname)
  cp.fig.savefig("%s.eps"%figname)
  print("Saved %s.pdf/eps"%figname)

def plot_compare(fitdf,models,figname="compare_path"):
  parmpath=models[-1].params.index
  figc,axc = plt.subplots(1,len(parmpath),sharex=True,sharey=True,squeeze=False)
  axc=axc[0]
  for ax,model in zip(axc,models): 
    ax.plot([fitdf['energy'].min()*ev,fitdf['energy'].max()*ev],[fitdf['energy'].min()*ev,fitdf['energy'].max()*ev],'k')
    ax.plot(fitdf['energy']*ev,model.fittedvalues*ev,'.',alpha=0.5)

  # Plot E vs E.
  print("Compare energies")
  axc[0].set_ylabel('model (eV)')
  axc[0].set_xticks(pt.thin_ticks(axc[0].get_xticks()))
  for parm,ax in zip(parmpath,axc):
    ax.set_title("Add %s"%parm)
    ax.set_xlabel('ab initio (eV)')
  figc.set_size_inches(2.5*axc.shape[0]+0.5,3)
  figc.tight_layout()
  figc.savefig(figname+'.pdf')
  figc.savefig(figname+'.eps')
  print("Saved %s.pdf/eps"%figname)


  #Plot final  E vs. E
  finaldf=fitdf.copy()
  finaldf['model']=models[-1].fittedvalues
  finaldf['type']='U'
  finaldf.loc[finaldf['id'].apply(lambda x:'r' in x),'type']='RO'
  finaldf[['model','energy','energy_err']]*=ev
  #cp=pt.CategoryPlot(finaldf,
  #    color='type',mark='excitation',
  #    mmap={'none':'o','singles':'s','doubles':'d'},
  #    cmap={'U':pt.pc['b'],'RO':pt.pc['r']})
  cp=pt.CategoryPlot(finaldf,
      color='excitation',mark='excitation',
      cmap={'none':pt.pc['b'],'singles':pt.pc['r'],'doubles':pt.pc['g']},
      mmap={'none':'o','singles':'s','doubles':'d'})
  cp.axes[0,0].plot([fitdf['energy'].min()*ev,fitdf['energy'].max()*ev],[fitdf['energy'].min()*ev,fitdf['energy'].max()*ev],'k')
  cp.axes[0,0].set_ylabel('Model Energy (eV)')
  cp.axes[0,0].set_xlabel(r'\textit{ab initio} Energy (eV)')
  args=copy(pt.make_plotargs())
  args['ms']=3
  cp.plot('energy','model',plotargs=args)
  cp.add_legend(args={'title':'Excitation','loc':'upper left','bbox_to_anchor':(0.9,0.8)})
  sns.despine()
  cp.fig.set_size_inches(4+1,3)
  cp.fig.subplots_adjust(right=0.8)
  cp.fig.savefig(figname+'_final.pdf')
  cp.fig.savefig(figname+'_final.eps')
  print("Saved %s.pdf/eps"%figname)

def mypca(df):
  # This can probably be made more efficient by using an SVD.
  # This carries the unbiased estimator, which my current sklearn doesn't have.
  cov=df.cov()
  eigval,eigvec=np.linalg.eigh(cov.values)
  order=(-eigval).argsort()
  eigval=eigval[order]
  eigvec=eigvec[order]

  return eigval,eigvec

def parallel(parmdf,ref):
  fig,axes=plt.subplots(1,1)
  desc=parmdf.columns
  
  # Copy and scale.
  pltdf=parmdf.copy()
  pltdf-=ref
  pltdf['energy']*=ev/5

  for idx,row in pltdf.iterrows():
    axes.plot(range(pltdf.shape[1]),row[desc])

  #print("Reference to descriptors.")
  #print(ref[desc])

  axes.set_xlabel('RDM sum')
  axes.set_ylabel('Value relative to ground')
  axes.set_xticks(range(pltdf.shape[1]))
  axes.set_xticklabels(pltdf.columns,rotation=45,ha='right',va='top')
  pt.fix_lims(axes)

  sns.despine()
  fig.set_size_inches(4,3)
  fig.tight_layout()
  fig.savefig('para.pdf')
  fig.savefig('para.eps')
  print("Saved para.pdf/eps.")

def compare_basis_desc():
  iaodf=pd.read_csv('iao.qmcdf.csv')
  iaodf=iaodf[~(abs(iaodf['totocc']-12)>0.5)]
  ugsdf=pd.read_csv('uksgs.qmcdf.csv')
  ugsdf=ugsdf[~(abs(ugsdf['totocc']-12)>0.5)]
  rgsdf=pd.read_csv('rksgs.qmcdf.csv')
  rgsdf=rgsdf[~(abs(rgsdf['totocc']-12)>0.5)]
  rmddf=pd.read_csv('rksmid.qmcdf.csv')
  rmddf=rmddf[~(abs(rmddf['totocc']-12)>0.5)]

  comdf=pd.merge(iaodf,rgsdf,
      on=['id','extid','excitation'],suffixes=('_iao','_other'))

  for col in comdf.columns:
    if 'iao' in col:
      other=col.replace('iao','other')
      comdf[col.replace('_iao','')]=abs(comdf[other]-comdf[col])
      comdf=comdf.drop([col,other],axis=1)
  print(comdf)
  fig,axes=plt.subplots(2,len(desc))
  fig.set_size_inches(30,6)
  axes[0,0].set_ylabel("IAO-basis values")
  axes[1,0].set_ylabel("Fixed-basis relative")
  for ax,col in zip(axes[0],desc):
    sns.distplot(iaodf[col], hist=False, rug=True, ax=ax)#, kde_kws={'bw':0.005})
    ax.set_xticks(pt.thin_ticks(ax.get_xticks()))
  for ax,col in zip(axes[1],desc):
    sns.distplot(comdf[col], hist=False, rug=True, ax=ax)#, kde_kws={'bw':0.005})
    ax.set_xticks(pt.thin_ticks(ax.get_xticks()))
  fig.tight_layout()
  fig.savefig('dist.pdf')
  fig.savefig('dist.eps')
  print("Saved dist.pdf/eps")

def check_correlation_coef():
  pd.set_option('display.max_columns', None, 'display.width', 200)
  fitdf,models=pkl.load(open('fitdf.pkl','rb'))
  resids=models[8].resid
  fitdf['Jxzyz']-=fitdf['Jxzyz'].mean()
  fitdf['cov_term']=fitdf['Jxzyz']*resids
  fitdf['sort']=-fitdf['cov_term'].abs()
  fitdf['energy']=(fitdf['energy']-fitdf['energy'].min())*ev
  print(fitdf.describe())
  for parm in desc:
    #fitdf[parm]/=fitdf[parm].abs().max()
    #fitdf[parm]-=fitdf[parm].mean()
    pass
  print(fitdf.sort_values('sort')[['id','extid','energy','cov_term']+desc])

def subset_samples_dimedge(nedge=1,ewin=8/ev,target=desc):
  ''' Find subsets of total samples and fit a model to them. Chooses points that on extreamal in each dimension.
  Args:
    nedge (int): Number of samples to include for each descriptor.
    ewin (float): Highest energy difference from the ground state to include in the sample.
  Returns:
    list: List of models from MP fitting procedure.
  '''
  pd.set_option('display.max_columns', None, 'display.width', 200)
  qmcdf=pd.read_csv('qmcdf.csv')

  # Some filtering of the database.
  outofspace=abs(qmcdf['totocc']-12)>0.5
  qmcdf=qmcdf[~outofspace].copy().reset_index(drop=True)
  high=qmcdf['energy']-qmcdf['energy'].min()>ewin
  qmcdf=qmcdf[~high].copy().reset_index(drop=True)

  # Generate database of differences from mean.
  diffdf=qmcdf.copy()
  cols=['energy','energy_err','variance','totocc']+target
  diffdf[cols]=qmcdf[cols].apply(lambda x: (x-x.mean())/x.mean())
  diffdf['euclidian']=diffdf[target].apply(lambda x:(x**2).sum()**0.5,axis=1)

  edgedf=pd.DataFrame(diffdf.sort_values('euclidian',ascending=True).iloc[:1][['id','extid']])

  # Find elements that stretch the farthest in each descriptor direction, and the middle.
  for d in target:
    edgedf=pd.concat([ edgedf,diffdf.sort_values(d,ascending=True).iloc[:nedge][['id','extid']] ])
    edgedf=pd.concat([ edgedf,diffdf.sort_values(d,ascending=False).iloc[:nedge][['id','extid']] ])

  edgedf=edgedf.drop_duplicates().merge(qmcdf,on=['id','extid'])

  return edgedf

def subset_samples_greedy(nsubset,ewin=8/ev,target=desc,startdf=None):
  ''' Find subsets of total samples and fit a model to them. Greedily chooses points
  that are far from previous subset of points.
  Args:
    nsubset (int): Number of samples to include for each descriptor.
    ewin (float): Highest energy difference from the ground state to include in the sample.
  Returns:
    list: List of models from MP fitting procedure.
  '''
  pd.set_option('display.max_columns', None, 'display.width', 200)
  qmcdf=pd.read_csv('qmcdf.csv')

  # Some filtering of the database.
  outofspace=abs(qmcdf['totocc']-12)>0.5
  qmcdf=qmcdf[~outofspace].copy().reset_index(drop=True)
  high=qmcdf['energy']-qmcdf['energy'].min()>ewin
  qmcdf=qmcdf[~high].copy().reset_index(drop=True)

  # Generate database of differences from mean, relative to SD.
  cendf=qmcdf.copy()
  cols=['energy','energy_err','variance','totocc']+target
  cendf[cols]=qmcdf[cols].apply(lambda x: (x-x.mean())/x.std())
  cendf['euclidian']=cendf[target].apply(lambda x:(x**2).sum()**0.5,axis=1)
  cendf['set']=0.0

  if startdf is None:
    # Start with a state close to the center of the space.
    edgedf=cendf[cendf['euclidian']==cendf['euclidian'].min()]
  else:
    edgedf=startdf.copy()

  def diff_set(row):
    total=row['set']
    row_in=edgedf.iloc[-1]
    total+=((row[target]-row_in[target])**2).sum()**0.5 

    return total

  # Greedy add points as far from previous set as possible.
  while edgedf.shape[0]<nsubset:
    cendf=cendf[~cendf.index.isin(edgedf.index)]
    cendf['set']=cendf[target+['set']].apply(diff_set,axis=1)
    edgedf=pd.concat([ edgedf, cendf[cendf['set']==cendf['set'].max()].copy()] )

  edgedf=edgedf[['id','extid']].drop_duplicates().merge(qmcdf,on=['id','extid'])
  return edgedf

def check_subsets(nsubsets=[20,30,40,50],ewins=[8/ev],target=['J','tsigd','Ud','tsigs','edxy','es']):
  ''' Scan across different subsets of the samples.
  Args:
    nsubsets (iterable): list of nsubset args.
    ewins (iterable): list of ewin args.
  '''
  if True: #not os.path.exists('moddf.pkl'):
    moddf=pd.DataFrame()
    for nsubset in nsubsets:
      for ewin in ewins:
        edgedf=subset_samples_greedy(nsubset,ewin,target)
        print(edgedf.shape)
        models=fit_procedure(edgedf,['E0']+target)
        res={'nsamples':[edgedf.shape[0]],'ewin':[ewin]}
        res.update(zip(range(len(models)),[[m] for m in models]))
        moddf=pd.concat([moddf,pd.DataFrame(res)])
    moddf=moddf.melt(id_vars=['nsamples','ewin'],var_name='MP step',value_name='model')
    pkl.dump(moddf,open('moddf.pkl','wb'))
  else:
    moddf=pkl.load(open('moddf.pkl','rb'))

  # Unpack parameter values.
  parmdf=moddf.join(moddf['model'].apply(lambda x:x.params)).drop('model',axis=1)
  parmdf=parmdf.fillna(0.0)
  parmdf=parmdf.melt(id_vars=['nsamples','ewin','MP step'],var_name='parm',value_name='value')

  # Unpack parameter errors.
  errdf=moddf.join(moddf['model'].apply(lambda x:x.bse)).drop('model',axis=1)
  errdf=errdf.fillna(0.0)
  errdf=errdf.melt(id_vars=['nsamples','ewin','MP step'],var_name='parm',value_name='error')

  # Combine and touchup.
  parmdf=parmdf.merge(errdf,on=['nsamples','ewin','MP step','parm'])
  parmdf[['value','error','ewin']]*=ev

  # Plot MP results for all subsets of data.
  cp=pt.CategoryPlot(parmdf[parmdf['parm']!='E0'].copy(),
      color='parm',mark='parm',row='ewin',col='nsamples',
      sharex=True,sharey=True)
  cp.plot('MP step','value','error',plotargs=pt.make_plotargs(),labrow=True,labcol=True)
  pt.fix_lims(cp.axes)
  sns.despine()

  cp.fig.savefig('subsets.pdf')
  cp.fig.savefig('subsets.eps')

  # Unpack full sampling.
  _,models=pkl.load(open('fitdf.pkl','rb'))
  bestparmdf=pd.DataFrame([m.params for m in models]).reset_index()
  besterrdf=pd.DataFrame([m.bse for m in models]).reset_index()
  bestparmdf=bestparmdf.melt(id_vars='index',var_name='parm')
  besterrdf=besterrdf.melt(id_vars='index',var_name='parm',value_name='error')
  bestparmdf.loc[bestparmdf['value'].isnull(),'value']=0.0
  besterrdf.loc[besterrdf['error'].isnull(),'error']=0.0
  bestparmdf=pd.merge(bestparmdf,besterrdf,on=['index','parm'],how='outer')
  bestparmdf[['value','error']]*=ev
  bestparmdf=bestparmdf.rename(columns={'index':'MP step'})

  # Add a col for best sampling. Take relative.
  parmdf=parmdf.merge(bestparmdf,how='outer',on=['MP step','parm'],suffixes=('','_best'))
  parmdf['value']-=parmdf['value_best']

  # Plot MP results for all subsets of data.
  cp=pt.CategoryPlot(parmdf[parmdf['parm']!='E0'].copy(),
      color='parm',mark='parm',row='ewin',col='nsamples',
      sharex=True,sharey=True)
  cp.plot('MP step','value',plotargs=pt.make_plotargs(),labrow=True,labcol=True)
  pt.fix_lims(cp.axes)
  sns.despine()

  cp.fig.savefig('subsets_rel.pdf')
  cp.fig.savefig('subsets_rel.eps')

def analysis():
  print("Special analysis")

  # Analyze previous fit. 
  fitdf,models=pkl.load(open("fitdf.pkl",'rb'))
  #plot_mses(fitdf,models)
  #plot_params(fitdf,models)
  #plot_corr(fitdf,models)
  plot_compare(fitdf,models)
  #plot_bic(fitdf,models)

  # Analyze sparse sampling
  #fitdf,models=pkl.load(open("sparse.fitdf.pkl",'rb'))
  #plot_mses(fitdf,models,figname="sparse.mse_path")
  #plot_params(fitdf,models,figname="sparse.parm_vals")
  #plot_corr(fitdf,models,figname="sparse.corr_path")
  #plot_compare(fitdf,models,figname="sparse.compare_path")
  #plot_bic(fitdf,models,figname="sparse.measure_path")

  # Analyze jlast fits.
  #fitdf,models=pkl.load(open("jlast.fitdf.pkl",'rb'))
  #plot_mses(fitdf,models,figname="jlast.mse_path")
  #plot_params(fitdf,models,figname="jlast.parm_vals")
  #plot_corr(fitdf,models,figname="jlast.corr_path")
  #plot_compare(fitdf,models,figname="jlast.compare_path")

  # Analyze MP fits.
  #fitdf,models=pkl.load(open("mp.fitdf.pkl",'rb'))
  #plot_mses(fitdf,models,figname="mp.mse_path")
  #plot_params(fitdf,models,figname="mp.parm_vals")
  #plot_corr(fitdf,models,figname="mp.corr_path")
  #plot_compare(fitdf,models,figname="mp.compare_path")
  #plot_bic(fitdf,models,figname="mp.measure_path")
  print("Done analysis.")

def plot_smaller_model(
    ewin=8/ev,
    target=['J','tsigd','Ud']
    ):
  qmcdf=pd.read_csv('qmcdf.csv')

  # Some filtering of the database.
  outofspace=abs(qmcdf['totocc']-12)>0.5
  qmcdf=qmcdf[~outofspace].copy().reset_index(drop=True)
  high=qmcdf['energy']-qmcdf['energy'].min()>ewin
  qmcdf=qmcdf[~high].copy().reset_index(drop=True)
  edgedf=subset_samples_greedy(40,ewin=8/ev,target=target)

  fullmodels=fit_procedure(qmcdf,['E0']+target)
  edgemodels=fit_procedure(edgedf,['E0']+target)

  plot_compare(qmcdf,fullmodels,figname='full_compare')
  plot_compare(edgedf,edgemodels,figname='edge_compare')
  plot_params(qmcdf,fullmodels,figname='full_params')
  plot_params(edgedf,edgemodels,figname='edge_params')
  plot_bic(qmcdf,fullmodels,figname='full_bic')
  plot_bic(edgedf,edgemodels,figname='edge_bic')

if __name__=='__main__':
  # Generates fit data.
  fit_FeSe()
  analysis()
  #subset_samples_greedy()
  #check_subsets()
  plot_smaller_model()

  # Debugging fits.
  #compare_all_iaos(pd.read_pickle('scfdf.pkl'))
  #compare_basis_desc()
  #check_correlation_coef()

