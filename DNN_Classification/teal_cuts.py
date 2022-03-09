import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#this script was used to help hard cut the data according to Teals' cuts, 
#keeping cb>0.4 as background and cb<-.4 as signal, regardless of if its with source or not

#it is used inside the plot_confusion_matrix.py script, to return the values of trueY and predicted and plot a confusion matrix using the existing code

def cuts():
    data = pd.read_csv("predicted_data.csv")
    data = pd.read_csv("predicted_data_evaluate.csv")
    #print(data.head())
    Sdf = data.loc[data['TrueY']==1]
    Bdf = data.loc[data['TrueY']==0]
        
    evts_sel_CBless0_4 = Sdf.loc[Sdf['5']<=0.4]
    print(evts_sel_CBless0_4.head(1))
    evts_sel_CBless0_4.loc[:,('TrueY')] = '1'
    evts_sel_CBless0_4.loc[:,('Predicted')] = '1'
    print(evts_sel_CBless0_4.head(1))
    
    bkgd_sel_CBless0_4 = Bdf.loc[Bdf['5']<=0.4]
    bkgd_sel_CBless0_4.loc[:,('TrueY')] = '0'
    bkgd_sel_CBless0_4.loc[:,('Predicted')] = '1'
    #print(evts_sel_CBless0_4[3000:3001], bkgd_sel_CBless0_4[3000:3001])
    
    
    evts_drop_CBmore0_4 = Sdf.loc[Sdf['5']>0.4]
    evts_drop_CBmore0_4.loc[:,('TrueY')] = '1'
    evts_drop_CBmore0_4.loc[:,('Predicted')] ='0'
    
    bkgd_drop_CBmore0_4 = Bdf.loc[Bdf['5']>0.4]
    bkgd_drop_CBmore0_4.loc[:,('TrueY')] ='0'
    bkgd_drop_CBmore0_4.loc[:,('Predicted')] = '0'
    
    print("With CB cut<0.4 we select ", evts_sel_CBless0_4.shape," signal events and ",bkgd_sel_CBless0_4.shape," Bkgd events.")
    print("With CB cut<0.4 we drop ", evts_drop_CBmore0_4.shape," signal events and",bkgd_drop_CBmore0_4.shape," Bkgd events.")
    ytrue = pd.concat((evts_sel_CBless0_4, bkgd_drop_CBmore0_4))
    ypred = pd.concat((evts_drop_CBmore0_4 ,bkgd_sel_CBless0_4))
    ytrue_pred = pd.concat((ytrue,ypred))
    return ytrue_pred
'''
data = pd.read_csv("predicted_data.csv")
data = pd.read_csv("predicted_data_evaluate.csv")
#print(data.head())
Sdf = data.loc[data['TrueY']==1]
Bdf = data.loc[data['TrueY']==0]
    
evts_sel_CBless0_4 = Sdf.loc[Sdf['5']<=0.4]
print(evts_sel_CBless0_4.head(1))
evts_sel_CBless0_4.loc[:,('TrueY')] = '1'
evts_sel_CBless0_4.loc[:,('Predicted')] = '1'
print(evts_sel_CBless0_4.head(1))

bkgd_sel_CBless0_4 = Bdf.loc[Bdf['5']<=0.4]
bkgd_sel_CBless0_4.loc[:,('TrueY')] = '0'
bkgd_sel_CBless0_4.loc[:,('Predicted')] = '1'
#print(evts_sel_CBless0_4[3000:3001], bkgd_sel_CBless0_4[3000:3001])


evts_drop_CBmore0_4 = Sdf.loc[Sdf['5']>0.4]
evts_drop_CBmore0_4.loc[:,('TrueY')] = '1'
evts_drop_CBmore0_4.loc[:,('Predicted')] ='0'

bkgd_drop_CBmore0_4 = Bdf.loc[Bdf['5']>0.4]
bkgd_drop_CBmore0_4.loc[:,('TrueY')] ='0'
bkgd_drop_CBmore0_4.loc[:,('Predicted')] = '0'

print("With CB cut<0.4 we select ", evts_sel_CBless0_4.shape," signal events and ",bkgd_sel_CBless0_4.shape," Bkgd events.")
print("With CB cut<0.4 we drop ", evts_drop_CBmore0_4.shape," signal events and",bkgd_drop_CBmore0_4.shape," Bkgd events.")
ytrue = pd.concat((evts_sel_CBless0_4, bkgd_drop_CBmore0_4))
ypred = pd.concat((evts_drop_CBmore0_4 ,bkgd_sel_CBless0_4))
ytrue_pred = pd.concat((ytrue,ypred))
'''

