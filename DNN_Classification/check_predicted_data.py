import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("predicted_data.csv")
print(data.head())
Sdf_prompt = data.loc[data['TrueY']==1]
Bdf = data.loc[data['TrueY']==0]

evts_sel_CBless0_4 = Sdf_prompt.loc[Sdf_prompt['3']<0.4]
bkgd_sel_CBless0_4 = Bdf.loc[Bdf['3']<0.4]
print("With CB cut<0.4 we select ", evts_sel_CBless0_4.shape," signal events and ",bkgd_sel_CBless0_4.shape," Bkgd events.")
evts_drop_CBmore0_4 = Sdf_prompt.loc[Sdf_prompt['3']>=0.4]
bkgd_drop_CBmore0_4 = Bdf.loc[Bdf['3']>=0.4]
print("With CB cut<0.4 we drop ", evts_drop_CBmore0_4.shape," signal events and",bkgd_drop_CBmore0_4.shape," Bkgd events.")

def Make2DHist(df,xvariable,yvariable,labels,ranges):
    plt.hist2d(df[xvariable],df[yvariable], bins=(ranges['xbins'],ranges['ybins']),
              range=[ranges['xrange'],ranges['yrange']],
              cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
 
'''
#--- CB to cluster Time:   
labels = {'title': 'Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,2000],'yrange':[0,1]}
Make2DHist(Sdf_prompt,'0','3',labels,ranges)
plt.show()
labels = {'title': 'Charge balance parameters in time window \n (Bkgd data)',
          'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,20000],'yrange':[0,1]}
Make2DHist(Bdf[:3570],'0','3',labels,ranges)
plt.show()

#--- CB to clusterPE:
labels = {'title': 'Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
Make2DHist(Sdf_prompt,'1','3',labels,ranges)
plt.show()
labels = {'title': 'Charge balance parameters in time window \n (Bkgd data)',
          'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
Make2DHist(Bdf[:3570],'1','3',labels,ranges)
plt.show()
'''
#----------plot data we keep: 
#--- CB to clusterPE:
TSig = Sdf_prompt.loc[(data['Predicted']==1)]
FSig = Bdf.loc[(data['Predicted']==1)]

labels = {'title': 'True signal: Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
Make2DHist(TSig,'1','3',labels,ranges)
plt.show()


labels = {'title': 'False signal: Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
Make2DHist(FSig,'1','3',labels,ranges)
plt.show()

#--- CB to cluster Time:   
labels = {'title': 'True signal: Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,20000],'yrange':[0,1]}
Make2DHist(TSig,'0','3',labels,ranges)
plt.show()
labels = {'title': 'False signal: Charge balance parameters in time window \n (Bkgd data)',
          'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,20000],'yrange':[0,1]}
Make2DHist(FSig[:3570],'0','3',labels,ranges)
plt.show()

#TSig-PMT IDs:
rest, IDs, y = np.split(TSig,[5,1500],axis=1)
ids = IDs.to_numpy()
Tsig_IDs = ids[ids != 0]
#print(type(Tsig_IDs))
#print(Tsig_IDs[:10])
plt.hist(Tsig_IDs, bins =150)
plt.xlim(320, 470)
plt.title("True signal: PMT IDs")
plt.xlabel('PMT IDs')
plt.show()
#FSig-PMT IDs:
rest1, IDs1, y1 = np.split(FSig,[5,1500],axis=1)
ids1 = IDs1.to_numpy()
Fsig_IDs = ids1[ids1 != 0]
plt.hist(Fsig_IDs, bins =150)
plt.xlim(320, 470)
plt.title("False signal: PMT IDs")
plt.xlabel('PMT IDs')
plt.show()

#----------plot data we treat as Bkgd: 
TBkgd = Bdf.loc[(data['Predicted']==0)]
FBkgd = Sdf_prompt.loc[(data['Predicted']==0)]

#--- CB to cluster Time:   
labels = {'title': 'True Bkgd: Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,20000],'yrange':[0,1]}
Make2DHist(TBkgd,'0','3',labels,ranges)
plt.show()

labels = {'title': 'False Bkgd: Charge balance parameters in time window \n (Bkgd data)',
          'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,20000],'yrange':[0,1]}
Make2DHist(FBkgd[:3570],'0','3',labels,ranges)
plt.show()

#--- CB to clusterPE:
labels = {'title': 'True Bkgd: Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
Make2DHist(TBkgd,'1','3',labels,ranges)
plt.show()

labels = {'title': 'False Bkgd: Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
          'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
Make2DHist(FBkgd,'1','3',labels,ranges)
plt.show()

#TBkgd -PMT IDs:
rest2, IDs2, y2 = np.split(TBkgd,[5,1500],axis=1)
ids2 = IDs2.to_numpy()
TBkgd_IDs = ids2[ids2 != 0]
plt.hist(TBkgd_IDs, bins =150)
plt.xlim(320, 470)
plt.title("True Bkgd: PMT IDs")
plt.xlabel('PMT IDs')
plt.show()

#FBkgd- PMT IDs:
rest3, IDs3, y3 = np.split(FBkgd,[5,1500],axis=1)
ids3 = IDs3.to_numpy()
FBkgd_IDs = ids3[ids3 != 0]
plt.hist(FBkgd_IDs, bins =150)
plt.xlim(320, 470)
plt.title("False Bkgd: PMT IDs")
plt.xlabel('PMT IDs')
plt.show()

