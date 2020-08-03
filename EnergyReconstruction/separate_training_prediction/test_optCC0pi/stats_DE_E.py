import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ROOT
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
plt.rc('font', family='Times New Roman', size=18)

infile = "Ereco_results.csv"

filein = open(str(infile))
print("number of events: ",filein)
df00=pd.read_csv(filein)
print("df00.head() ",df00.head())
Etrue = df00['MuonEnergy']
Ereco = df00['RecoE']

DE_E = 100.*abs((Etrue-Ereco)/Etrue)
#print(DE_E)
#---- EMu ----
_=plt.figure(0)
x = np.sort(DE_E)
y = np.arange(1, len(x)+1) / len(x)
#_ = plt.plot(x, y, color='red',label='Muon')#marker='.', linestyle='none')
l1, = plt.plot(x, y, color='red',label='Muon')
_ = plt.xlabel('$|\Delta E|/E$ [%]')
_ = plt.ylabel('Cumulative Distribution')
_=plt.xlim(0,30)
_=plt.ylim(0.,1.)
xl1=np.arange(0,4.45,0.01)
yl1=np.array(0.68-xl1+xl1)
yl0=np.arange(0,0.68,0.01)
xl0=np.array(4.45 -yl0 +yl0)
_=plt.plot(xl1, yl1, color='lightgray')
_=plt.plot(xl0,yl0, color='lightgray')
_= plt.legend(prop={'size': 12},handles=[l1], loc='center right')
plt.show()
#plt.savefig("DE_E_ECDFMU.png")
 

