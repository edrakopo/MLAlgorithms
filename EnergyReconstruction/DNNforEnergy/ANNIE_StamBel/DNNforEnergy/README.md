DNN for Energy Reco: 
These files are written for a DNN algorithm using both "hits" and "digits" variables contained in the "tankPMT_forEnergy_0*.root" (old) and "tankPMT_forEnergy_bigfile.root" files respectively (new). 
The files that we use have been first processed by ToolChain "ToolAanalysisLink" using "PhaseIITreeMaker Tool".
First we create a .csv from the .root files with "tankCreatecsv.C". 
We use the .csv of the old files and some of the data from the .csv of the new files for training. -->"DNNEnergy_Keras_train.py" 
We use the rest of the data from the .csv of the new files for prediction. -->"DNNEnergy_Keras_pred.py" 
Then, we run "plots_Energy.py" and "make_plots_fromcsv.C" for the graphs.


We observed the fact that we cannot make a good prediction for Energy using both "hits" and "digits". We will continue with a DNN based only on "digits".

The files containing "*_till1000*" are based on data for energies till 1000Mev.
The files containing "*_till5000*" or NaN are based on data for all energies.
