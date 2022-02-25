import numpy as np

def ReadHistFile(fi):
    with open(fi,"r") as f:
        lines = f.readlines()
    histline = None
    reading_bins = False
    bin_content = {}
    for j,line in enumerate(lines):
        if line.find("BINS")!=-1:
            histline = lines[j+1].rstrip("\n").split(",")
            for k,h in enumerate(histline):
                histline[k] = int(h)
            continue
        if line.find("BIN_CONTENT")!=-1:
            reading_bins = True
            continue
        if reading_bins:
            binline = line.split(",")
            bin_content[int(binline[0])] = int(binline[1].rstrip("\n"))
    print("BIN LINE: " + str(histline))
    print("BIN CONTENT DICT: " + str(bin_content))
    bin_lefts = np.arange(histline[0],histline[1], (histline[1] - histline[0])/float(histline[2]))
    bin_zeros = np.zeros(histline[2])
    for b in bin_content:
        bin_zeros[b]+=bin_content[b]
    return bin_zeros,bin_lefts

def ReadHistFileSimple(fi):
    with open(fi,"r") as f:
        lines = f.readlines()
    bin_lefts = []
    bins = []
    for j,line in enumerate(lines):
        binline = line.split(",")
        bin_lefts.append(float(binline[0]))
        bins.append(float(binline[1].rstrip("\n")))
    return np.array(bins),np.array(bin_lefts)

