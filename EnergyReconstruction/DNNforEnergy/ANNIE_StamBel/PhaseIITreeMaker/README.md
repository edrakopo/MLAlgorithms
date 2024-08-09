PhaseIITreeMaker Tool:
Here are the changes that we made in "PhaseIITreeMaker" Tool so we can extract the variables "digits" from it and use them for later analysis.

The initialization of the variables "NDigitsLAPPDs" and "NDigitsPMTs" is not correct. The variables are -9999 from their original values. 
It either needs to be changed or the changes must be done later when creating a .csv for example.
In the DNN algorithms that we are using the "tankCreatecsv.C" algorithm is performing this change, so we can get the true values.

Some of the cuts in the configfile are interdependent. Be carefull when using them.
