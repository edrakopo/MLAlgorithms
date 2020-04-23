import pandas as pd
import numpy as np 

#Select verbosity! 
verbose=0

#--- Read data from RingEvents.csv
infile = "data/RingEvents.csv"
filein = open(str(infile))
print("data in: ",filein)
df0=pd.read_csv(filein, names=["Detector type","Radius in m","Angle in rad","Height in m","x in m","y in m","z in m","Ring number"])
print(df0.head())
#print(df0[210:220])
print("index vals: ",df0.index.values," Columns names: ",df0.columns.values)

#--- group per event: 
index_evt = df0.index[df0['Detector type'] == 'Event number'].tolist()
if verbose==1:
   print(len(index_evt), " index_evt: ",index_evt)
for i in range(2): #len(index_evt)):
    if verbose==1:
       print("event:", i ," index:",index_evt[i])
    if index_evt[i]==index_evt[-1]:
       dfs = df0[index_evt[i]:]
    else:
       dfs = df0[index_evt[i]:index_evt[i+1]]   
    #print("dfs: ",dfs)
    #--- drop these lines:
    skip_lines = dfs.index[dfs['Detector type'] == 'Event number'].tolist() + dfs.index[dfs['Detector type'] == 'Detector type'].tolist()
    #print("skip lines: ",dfs.index[dfs['Detector type'] == 'Event number'].tolist())
    df_evt = dfs.drop(skip_lines)
    print("df_evt: ",df_evt)
    
    #drop some columns from the data sample
    #X  = df_evt.drop(["Detector type", "Ring number", "Radius in m","Angle in rad","Height in m"], axis=1).values
    #X0  = df_evt.drop(["Detector type", "Ring number", "Radius in m","Angle in rad","Height in m"], axis=1)
    X0  = df_evt.drop(["Detector type", "Ring number"], axis=1)
    X0['Radius in m'] = X0['Radius in m'].astype('float64')
    X0['Angle in rad'] = X0['Angle in rad'].astype('float64')
    X0['Height in m'] = X0['Height in m'].astype('float64')
    X0['x in m'] = X0['x in m'].astype('float64')
    X0['y in m'] = X0['y in m'].astype('float64')
    X0['z in m'] = X0['z in m'].astype('float64')
    X = X0.values.flatten()
    print("type(X): ",type(X)," type(X0) ",type(X0))
    print("X.shape: ",X.shape) 
    print("X: ", X)
    #print("Columns names @ X: ",X0.columns.values)
    #if i==1:
    #   print(type(X[0])," ",X)
    if verbose==1: 
       print("X.shape: ",X.shape," type(X)",type(X)," len(X) ", len(X))
    #store true labels
    labels_true = df_evt["Ring number"].values
    nrings = max(labels_true)
    if verbose==1: 
       print("labels_true.shape: ",labels_true.shape," type(labels_true): ",type(labels_true))
       print("______________ New Event:", i ," index: ",index_evt[i]," True Number of Rings: ", nrings,"______________")
    Y = labels_true.astype('int')
   
    if index_evt[i]==0:
       print("It's the first event!")
       X_save = pd.DataFrame(X)
    else:
       print("X_save: ",X_save)
       X_save.append(pd.DataFrame(X), ignore_index = True)       
       X = X0.values.flatten()
    #X_save.to_csv("test.out", float_format = '%.3f')
    #total_arr = np.append( X, axis = 0)    
    #np.savetxt('test.out', X, delimiter=',',newline=" ") 
     

 
