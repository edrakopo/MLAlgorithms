import numpy as np
import scipy as sp

class ProfileLikelihoodBuilder(object):
    '''
    Class is used to build likelihood profiles given a signal and background
    hypothesis.  For this class, the signal is assumed to follow a Bernoulli distribution
    while the background is a Poisson distribution with some mean.
    '''
    def __init__(self):
        self.likelihood_function = None
        self.bkg_pois_mean = None
        self.bkg_pois_mean_unc = None

    def SetBkgMean(self,mean):
        self.bkg_pois_mean = mean

    def SetBkgMeanUnc(self,mean_unc):
        self.bkg_pois_mean_unc = mean_unc

    def BuildLikelihoodProfile(self,ProfileVariable,SignalDistribution,SignalDistribution_unc,randShoots,BackgroundDistribution,BkgDistUnc = None):
        '''
        Returns a Chi-squared profile given the input profile variables, the background mean neutron rate
        defined with the SetBkgMean method, and a signal distribution.
        '''
        if self.bkg_pois_mean is None:
            print("You have to set your background distribution's mean neutron count per window!")
            return None
        ChiSquares = []
        LowestChiSq = 1E12
        LowestChiSqProfile = None
        for var in ProfileVariable:
            #MCProfile = self.BuildMCProfile(var,SignalDistribution,randShoots)
            MCProfile = self.BuildMCProfileBkgDist(var,SignalDistribution,BackgroundDistribution,randShoots,BkgDistUnc)
            ChiSquare = self.CalcChiSquare(MCProfile,SignalDistribution,SignalDistribution_unc,BkgDistUnc)
            if ChiSquare<LowestChiSq:
                LowestChiSqProfile = MCProfile
                LowestChiSq = ChiSquare
            ChiSquares.append(ChiSquare)
        return ChiSquares, LowestChiSqProfile

    def BuildMCProfile(self,variable,SignalDistribution,randShoots):
        '''
        Given the probability of returning a 1 in the signal distribution and the defined background mean in
        self.bkg_pois_mean, build a Monte Carlo-based data distribution.
        '''
        Profile = np.zeros(len(SignalDistribution))
        poisson_means = np.random.normal(self.bkg_pois_mean,self.bkg_pois_mean_unc,randShoots)
        poisson_means = poisson_means[np.where(poisson_means>0)[0]]
        randShoots = len(poisson_means)
        neutron_counts = np.zeros(randShoots)
        randos = np.random.random(randShoots)
        saw_neutrons = np.where(randos<=variable)[0]
        neutron_counts[saw_neutrons]+=1
        Bkg_shoots = np.random.poisson(poisson_means)
        #Bkg_shoots = np.random.poisson(self.bkg_pois_mean,randShoots)
        neutron_counts = neutron_counts + Bkg_shoots
        Profile,Profile_edges = np.histogram(neutron_counts,range=(0,len(SignalDistribution)),bins=len(SignalDistribution))
        Profile_normed = Profile/np.sum(Profile)
        return Profile_normed

    def BuildMCProfileBkgDist(self,variable,SignalDistribution,BkgDistribution,randShoots,BkgDistUnc=None):
        '''
        Given the probability of returning a 1 in the signal distribution and the defined background mean in
        self.bkg_pois_mean, build a Monte Carlo-based data distribution.
        '''
        Profile = np.zeros(len(SignalDistribution))
        Bin_values = np.arange(0,len(SignalDistribution),1)
        neutron_counts = np.zeros(int(randShoots))
        neutron_counts += np.random.choice(Bin_values, int(randShoots),p=BkgDistribution)
        randos = np.random.random(int(randShoots))
        saw_neutrons = np.where(randos<=variable)[0]
        neutron_counts[saw_neutrons]+=1
        Profile,Profile_edges = np.histogram(neutron_counts,range=(0,len(SignalDistribution)),bins=len(SignalDistribution))
        Profile_normed = Profile/np.sum(Profile)
        return Profile_normed

    def CalcChiSquare(self,MCProfile,SignalDistribution,SignalDistribution_unc,BkgDistUnc):
        '''
        Given a hypothesis profile (MCProfile), calculate the chi-square relative to the
        input signal distribution.
        '''
        return np.sum(((MCProfile-SignalDistribution)/np.sqrt(SignalDistribution_unc**2 + BkgDistUnc**2))**2)


class ProfileLikelihoodBuilder2D(object):
    '''
    Class is used to build likelihood profiles given a signal and background
    hypothesis.  For this class, the signal is assumed to follow a Bernoulli distribution
    while the background is a Poisson distribution with some mean.
    '''
    def __init__(self):
        self.likelihood_function = None
        self.xprofile = None
        self.yprofile = None

    def SetEffProfile(self,xpro):
        self.xprofile = xpro

    def SetBkgMeanProfile(self,ypro):
        self.yprofile = ypro


    def BuildLikelihoodProfile(self,SignalDistribution,SignalDistribution_unc,randShoots):
        '''
        Returns a Chi-squared profile given the input profile variables, the background mean neutron rate
        defined with the SetBkgMean method, and a signal distribution.
        '''
        if self.xprofile is None or self.yprofile is None:
            print("Please define xprofile variables and yprofile variables")
            return None
        x_variable = []
        y_variable = []
        ChiSquares = []
        LowestChiSq = 1E12
        LowestChiSqProfile = None
        for xvar in self.xprofile:
            for yvar in self.yprofile:
                x_variable.append(xvar)
                y_variable.append(yvar)
                MCProfile = self.BuildMCProfile(xvar,yvar,SignalDistribution,randShoots)
                ChiSquare = self.CalcChiSquare(MCProfile,SignalDistribution,SignalDistribution_unc)
                if ChiSquare<LowestChiSq:
                    LowestChiSqProfile = MCProfile
                    LowestChiSq = ChiSquare
                ChiSquares.append(ChiSquare)
        return np.array(x_variable),np.array(y_variable),np.array(ChiSquares), LowestChiSqProfile

    def BuildMCProfile(self,neu_eff,bkg_rate,SignalDistribution,randShoots):
        '''
        Given the neutron capture efficiency and mean neutron multiplicity of the
        uncorrelated bkg component, build a Monte Carlo-based data distribution.
        '''
        neutron_counts = np.zeros(int(randShoots))
        randos = np.random.random(int(randShoots))
        saw_neutrons = np.where(randos<=neu_eff)[0]
        neutron_counts[saw_neutrons]+=1
        Bkg_shoots = np.random.poisson(bkg_rate,int(randShoots))
        #Bkg_shoots = np.random.poisson(self.bkg_pois_mean,randShoots)
        neutron_counts = neutron_counts + Bkg_shoots
        Profile,Profile_edges = np.histogram(neutron_counts,range=(0,len(SignalDistribution)),bins=len(SignalDistribution))
        Profile_normed = Profile/np.sum(Profile)
        return Profile_normed

    def CalcChiSquare(self,MCProfile,SignalDistribution,SignalDistribution_unc):
        '''
        Given a hypothesis profile (MCProfile), calculate the chi-square relative to the
        input signal distribution.
        '''
        return np.sum(((MCProfile-SignalDistribution)/SignalDistribution_unc)**2)
