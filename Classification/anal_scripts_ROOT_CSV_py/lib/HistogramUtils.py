import numpy as np

def AggregateBins(orig_bins, orig_bin_lefts,num_bins,bin_min,bin_max):
    NewBin_lefts = np.arange(bin_min,bin_max+((bin_max-bin_min)/(num_bins+1)),(bin_max-bin_min)/(num_bins))
    NewBins = np.zeros(len(NewBin_lefts))
    for k,entry in enumerate(orig_bin_lefts):
        print("ORIGINAL BIN LEFT: " + str(entry))
        if entry>np.max(NewBin_lefts):
            break
        if entry <np.min(NewBin_lefts):
            continue
        for j,val in enumerate(NewBin_lefts):
            print("NEW BIN LEFT: " + str(val))
            if val <= entry:
                continue
            else:
                print("FILLING NEW BIN " + str(j))
                NewBins[j]+=orig_bins[k]
                break
    NewBin_lefts = NewBin_lefts[0:len(NewBin_lefts)-1] 
    NewBins = NewBins[0:len(NewBins)-1] 
    return NewBins, NewBin_lefts
