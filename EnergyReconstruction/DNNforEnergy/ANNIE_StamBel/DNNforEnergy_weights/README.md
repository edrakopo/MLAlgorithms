DNN training and prediction algorithms based on the old files (vtxreco-beamlikemrd.root), including "hits" variables.
We use weights to provide a flat spectrum response on the events per energy bin.

**Note: Just adding the weights to the original algorithms did not work (original algorithms: /DNNforEnergy_old).
There was either a problem when shuffling or when calling the function and saving the model.
