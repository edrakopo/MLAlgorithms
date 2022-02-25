import uproot
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set_context('poster')
import sys
import numpy as np


def XZ_ToTheta(xdata, zdata):
    theta = []
    for i in range(len(xdata)):
        thistheta = None
        x = xdata[i]
        z = zdata[i]
        isnegative = False
        if x <0:
            isnegative = True
            x = -x
        r = np.sqrt(z**2 + x**2)
        if r == 0:
            thistheta = np.arccos(0)*180.0/np.pi
        else:
            thistheta = np.arccos(z/r)*180.0/np.pi 
        if isnegative:
            thistheta = (360.0 - thistheta)
            #thistheta = (180.0 + thistheta)
        #Now, do the transormation to beam theta
        if thistheta < 180:
            thistheta = - thistheta
        else:
            thistheta =  (360.0 - thistheta)

        theta.append(thistheta) #yeah that's confusing labeling
    return np.array(theta)

def YVSTheta(df, entrynum,cdatatype,cdata_label,sum_duplicates=False,hitrange=None,title=None):
    xdata = np.array(df['hitX'][entrynum])
    ydata = np.array(df['hitY'][entrynum])
    zdata = np.array(df['hitZ'][entrynum])
    idata = np.array(df['hitDetID'][entrynum])
    cdata = np.array(df[cdatatype][entrynum])
    comb_xdata = []
    comb_ydata = []
    comb_zdata = []
    comb_cdata = []
    if sum_duplicates:
        #sum cdata of all hits for single IDs
        id_set = set(idata)
        for theid in id_set:
            thisid_ind = np.where(idata==theid)[0]
            print("NUM HITS FOR ID %s: %s"%(str(theid),str(len(thisid_ind))))
            comb_cdata.append(np.sum(cdata[thisid_ind]))
            comb_xdata.append(xdata[thisid_ind[0]])
            comb_ydata.append(ydata[thisid_ind[0]])
            comb_zdata.append(zdata[thisid_ind[0]])
        xdata = np.array(comb_xdata)
        ydata = np.array(comb_ydata)
        zdata = np.array(comb_zdata)
        cdata = np.array(comb_cdata)

    barrel = np.where(abs(ydata)<1.45)[0]
    tophits = np.where(ydata>1.45)[0]
    bothits = np.where(ydata<-1.45)[0]

    xtop = xdata[tophits]
    xbot = xdata[bothits]
    xbar = xdata[barrel]
    ytop = ydata[tophits]
    ybot = ydata[bothits]
    ybar = ydata[barrel]
    ztop = zdata[tophits]
    zbot = zdata[bothits]
    zbar = zdata[barrel]
    ctop = cdata[tophits]
    cbot = cdata[bothits]
    cbar = cdata[barrel]

    r = 1.0

    ztop = -ztop
    xtop = -xtop
    ztop = ztop + 1.45 + r
    xbot = -xbot
    zbot = zbot -1.45 - r

    fig = plt.figure()
    ax = fig.add_subplot(111)
    display_y = np.array(ybar)
    display_x = XZ_ToTheta(xbar,zbar)
    display_x = display_x * (np.pi * r/180)
    display_c = cbar

    display_x = np.concatenate((display_x,xtop))
    display_y = np.concatenate((display_y,ztop))
    display_c = np.concatenate((display_c,ctop))
    display_x = np.concatenate((display_x,xbot))
    display_y = np.concatenate((display_y,zbot))
    display_c = np.concatenate((display_c,cbot))


    #rect = patches.Rectangle((-180,-1.45),360,2.9,facecolor='black',zorder=0)
    rect = patches.Rectangle((-np.pi*r,-1.45),2*np.pi*r,2.9,facecolor='black',zorder=0)
    topcir = patches.Circle((0,2.45),r,facecolor='black',zorder=0)
    botcir = patches.Circle((0,-2.45),r,facecolor='black',zorder=0)
    ax.add_patch(rect)
    ax.add_patch(topcir)
    ax.add_patch(botcir)
    if hitrange is not None:
        sc = plt.scatter(display_x,display_y,c=display_c,s=230, marker='o',cmap=cm.jet,vmin=hitrange[0], vmax=hitrange[1])
    else: 
        sc = plt.scatter(display_x,display_y,c=display_c,s=230, marker='o',cmap=cm.jet)
    cbar = plt.colorbar(sc,label=cdata_label)
    cbar.set_label(label=cdata_label, size=30)
    cbar.ax.tick_params(labelsize=30)
    leg = ax.legend(loc=2)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    ax.set_xlabel("Barrel position (m)", fontsize=34)
    ax.set_ylabel("Vertical position (m)", fontsize=34)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(30)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(30)
    if title is None:
        plt.title(cdata_label,fontsize=34)
    else:
        plt.title(title,fontsize=34)
    plt.show()
   
def YVSTheta_Nhit(df, entrynum,hitrange=None,title=None):
    xdata = np.array(df['hitX'][entrynum])
    ydata = np.array(df['hitY'][entrynum])
    zdata = np.array(df['hitZ'][entrynum])
    idata = np.array(df['hitDetID'][entrynum])
    comb_xdata = []
    comb_ydata = []
    comb_zdata = []
    comb_cdata = []
    #sum cdata of all hits for single IDs
    id_set = set(idata)
    for theid in id_set:
        thisid_ind = np.where(idata==theid)[0]
        print("NUM HITS FOR ID %s: %s"%(str(theid),str(len(thisid_ind))))
        comb_cdata.append(len(thisid_ind))
        comb_xdata.append(xdata[thisid_ind[0]])
        comb_ydata.append(ydata[thisid_ind[0]])
        comb_zdata.append(zdata[thisid_ind[0]])
    xdata = np.array(comb_xdata)
    ydata = np.array(comb_ydata)
    zdata = np.array(comb_zdata)
    cdata = np.array(comb_cdata)

    barrel = np.where(abs(ydata)<1.45)[0]
    tophits = np.where(ydata>1.45)[0]
    bothits = np.where(ydata<-1.45)[0]

    xtop = xdata[tophits]
    xbot = xdata[bothits]
    xbar = xdata[barrel]
    ytop = ydata[tophits]
    ybot = ydata[bothits]
    ybar = ydata[barrel]
    ztop = zdata[tophits]
    zbot = zdata[bothits]
    zbar = zdata[barrel]
    ctop = cdata[tophits]
    cbot = cdata[bothits]
    cbar = cdata[barrel]

    r = 1.0

    ztop = -ztop
    xtop = -xtop
    ztop = ztop + 1.45 + r
    xbot = -xbot
    zbot = zbot -1.45 - r

    fig = plt.figure()
    ax = fig.add_subplot(111)
    display_y = np.array(ybar)
    display_x = XZ_ToTheta(xbar,zbar)
    display_x = display_x * (np.pi * r/180)
    display_c = cbar

    display_x = np.concatenate((display_x,xtop))
    display_y = np.concatenate((display_y,ztop))
    display_c = np.concatenate((display_c,ctop))
    display_x = np.concatenate((display_x,xbot))
    display_y = np.concatenate((display_y,zbot))
    display_c = np.concatenate((display_c,cbot))


    #rect = patches.Rectangle((-180,-1.45),360,2.9,facecolor='black',zorder=0)
    rect = patches.Rectangle((-np.pi*r,-1.45),2*np.pi*r,2.9,facecolor='black',zorder=0)
    topcir = patches.Circle((0,2.45),r,facecolor='black',zorder=0)
    botcir = patches.Circle((0,-2.45),r,facecolor='black',zorder=0)
    ax.add_patch(rect)
    ax.add_patch(topcir)
    ax.add_patch(botcir)
    if hitrange is not None:
        sc = plt.scatter(display_x,display_y,c=display_c,s=230, marker='o',cmap=cm.jet,vmin=hitrange[0], vmax=hitrange[1])
    else: 
        sc = plt.scatter(display_x,display_y,c=display_c,s=230, marker='o',cmap=cm.jet)
    cbar = plt.colorbar(sc,label="Hit count")
    cbar.set_label(label="Hit count", size=30)
    cbar.ax.tick_params(labelsize=30)
    leg = ax.legend(loc=2)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    ax.set_xlabel("Barrel position (m)", fontsize=34)
    ax.set_ylabel("Vertical position (m)", fontsize=34)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(30)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(30)
    if title is None:
        plt.title("Number of hits",fontsize=34)
    else:
        plt.title(title,fontsize=34)
    plt.show()

def XVSZ(df, entrynum,cdatatype,cdata_label,hitrange=None,title=None):
    xdata = df['hitX'][entrynum]
    ydata = df['hitY'][entrynum]
    zdata = df['hitZ'][entrynum]
    cdata = df[cdatatype][entrynum]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = np.array(ydata)
    theta = XZ_ToTheta(xdata,zdata)
    if hitrange is not None:
        sc = plt.scatter(xdata,zdata,c=cdata,s=200, marker='o',cmap=cm.jet,vmin=hitrange[0], vmax=hitrange[1])
    else: 
        sc = plt.scatter(xdata,zdata,c=cdata,s=200, marker='o',cmap=cm.jet)
    cbar = plt.colorbar(sc,label=cdata_label)
    cbar.set_label(label=cdata_label, size=30)
    cbar.ax.tick_params(labelsize=30)
    leg = ax.legend(loc=2)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    ax.set_xlabel("X (m)", fontsize=34)
    ax.set_ylabel("Z (m)", fontsize=34)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(30)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(30)
    if title is None:
        plt.title(cdata_label,fontsize=34)
    else:
        plt.title(title,fontsize=34)
    plt.show()
    
def XVSZ_barrel(df, entrynum,cdatatype,cdata_label,hitrange=None,title=None):
    xdata = np.array(df['hitX'][entrynum])
    ydata = np.array(df['hitY'][entrynum])
    zdata = np.array(df['hitZ'][entrynum])
    cdata = np.array(df[cdatatype][entrynum])
    barrel = np.where(abs(ydata)<1.45)[0]
    xdata=xdata[barrel]
    zdata=zdata[barrel]
    cdata=cdata[barrel]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if hitrange is not None:
        sc = plt.scatter(xdata,zdata,c=cdata,s=200, marker='o',cmap=cm.jet,vmin=hitrange[0], vmax=hitrange[1])
    else: 
        sc = plt.scatter(xdata,zdata,c=cdata,s=200, marker='o',cmap=cm.jet)
    cbar = plt.colorbar(sc,label=cdata_label)
    cbar.set_label(label=cdata_label, size=30)
    cbar.ax.tick_params(labelsize=30)
    leg = ax.legend(loc=2)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    ax.set_xlabel("X (m)", fontsize=34)
    ax.set_ylabel("Z (m)", fontsize=34)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(30)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(30)
    if title is None:
        plt.title(cdata_label,fontsize=34)
    else:
        plt.title(title,fontsize=34)
    plt.show()
    

