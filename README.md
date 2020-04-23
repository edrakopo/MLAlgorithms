--------------------------------------------- **Machine Learning Algorithms** ---------------------------------------------

Description of each directory: 

**Classification:** Scripts to classify different events. These are used to discriminate between muons and electrons in the ANNIE experiment. Project includes: [script1](https://github.com/edrakopo/MLAlgorithms/blob/master/Classification/classification_e_mu.py) to compare the performance of different classification algorithms (accuracy/ROC curves), [script2](https://github.com/edrakopo/MLAlgorithms/blob/master/Classification/optimising_parameters.py) for the optimisation of each algorithm parameters, [script3](https://github.com/edrakopo/MLAlgorithms/blob/master/Classification/plot_confusion_matrix.py) to plot the normalised confusion matrix.   

**Clustering:** Scripts to find the number of clusters - These are to be used to find the number of observed rings in the ANNIE experiment.

**DSG_Turing:** example scripts used for the NATS project (Data Study Group - Alan Turing Institute) to predict the aircraft trajectory using a [**ParticleFilter**](https://en.wikipedia.org/wiki/Particle_filter). See [code](https://github.com/edrakopo/MLAlgorithms/blob/master/DSG_Turing/ParticleFilter_aircraft.py). 

**EnergyReconstruction (Regression projects):** Scripts used to predict the track length and the particle energy in water Cherenkov detectors. 
- For the track length reconstruction in the ANNIE experiment we use a **Deep Learning Neural Network from Tensorflow**: See [code1](https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/TrackLengthReconstruction/DNNFindTrackLengthInWater_Keras_train.py) for training, [code2](https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/TrackLengthReconstruction/DNNFindTrackLengthInWater_Keras_pred.py) for prediction, [code for optimisation](https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/TrackLengthReconstruction/DNNFindTrackLengthInWater_Keras_optimise.py), [example data](https://github.com/edrakopo/MLAlgorithms/tree/master/EnergyReconstruction/data) and [paper](https://arxiv.org/pdf/1803.10624.pdf)
- For the muon/neutrino energy reconstruction in the ANNIE experiment we use a **BDTG from Scikit-Learn**. See [code1](https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/BDT_MuonEnergyReco.py), [code2](https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/BDT_NeutrinoEnergyReco.py), [example data](https://github.com/edrakopo/MLAlgorithms/tree/master/EnergyReconstruction/data) and [paper](https://arxiv.org/pdf/1803.10624.pdf). 
Such code can be trained in a different step. In this case, we train the algorithm and store the weights using [script1]( https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/separate_training_prediction/BDT_MuonEnergyReco_train.py) and we make the prediction using the existing weights and [script2](https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/separate_training_prediction/BDT_MuonEnergyReco_pred.py). To optimise the training parameters use [script](https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/optimisation/optimising_parameters_energy.py).
- Developed a new generic method to reconstruct the incident particle energy from observable data in Water-Cherenkov detectors. For this project, the **BDTG from Scikit-Learn** was found to show the best performance as documented in this [paper](https://arxiv.org/pdf/1710.05668.pdf)/[JINST 13 P04009](https://iopscience.iop.org/article/10.1088/1748-0221/13/04/P04009/pdf). See [code]( https://github.com/edrakopo/MLAlgorithms/blob/master/EnergyReconstruction/WCh_det_scikit_regression_energy_reco.py). The different codes that were tested: the **gradient BDT** algorithms from the Scikit-Learn 0.18.2 and TMVA packages (ROOT 5.34/23), a **multi-layer percepton Neural Network (NN)** from the TMVA package and a **multi-layer NN implemented using TensorFlow (TNN)** via the Keras 1.2.2 library in Python can be found [here](https://github.com/edrakopo/MLAlgorithms_def/tree/master/EnergyReconstruction/WChDet_energyReocnstruction).

**mini_projects:** mini projects for **NLP**, **Speech**, **Credit Risk** and **Time Series** analysis. 


