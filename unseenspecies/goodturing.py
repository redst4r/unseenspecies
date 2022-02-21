import numpy as np
from scipy.stats import binom
import pandas as pd
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.robjects


def custom_log(x, base):
    return np.log(x) / np.log(base)


def good_turing(t, prev: dict):
    """
    original Good-Turing estimator, unstable for t>1
    :param t: ratio of the number of future (m) and past samples (n), t=m/n
    :param prev: dict of prevalences
    """
    assert t<=1, "GT is unstable for t>1"

    U_GT = 0
    for i, freq in prev.items():
        U_GT += (-t)**i * freq

    U_GT = - U_GT
    return U_GT


def smoothed_good_turing(t, prev: dict):
    """
    based on
    Orlitsky et al. 2016:
    Estimating the number of unseen species: A bird in the hand is worth log n in the bush

    :param t: ratio of the number of future (m) and past samples (n), t=m/n
    :param prev: dict of prevalences
    #TODO :param smoothing_cdf: distribution function fo the smoother
    """

    assert 0 not in prev

    n = sum(i*j for i, j in prev.items())

    # determine the best parameters for the smoother
    logconstant = custom_log((n * t**2)/(t-1), 3)
    k = int(np.floor(0.5 * logconstant))
    q = 2/(t+2)

    # the ET smoothing (which is worse)
    # logconstant =  np.log2((n*t**2)/(t-1))
    # k = int(np.floor(0.5 * logconstant))
    # q = 1/ (t+1)

    smoothing_cdf = binom(k, q).cdf

    if True:
        U_L = 0
        for i, phi in prev.items():
            PL = 1-smoothing_cdf(i-1) # since P(L>=i) = 1-Cdf(L<=i-1)

            if PL > 0: # this is for numeric stability; for large i, PL is often 0, but t**i is enormous
                U_L += (-t)**i * phi * PL
            else:
                pass # PL == 0 mean no update to U_L
        U_L = - U_L
    else:
        U = []
        for i, phi in prev.items():
            PL = 1-smoothing_cdf(i-1) # since P(L>=i) = 1-Cdf(L<=i-1)

            if PL > 0:
                # this is for numeric stability; for large i, PL is often 0, but t**i is enormous
                U.append((-t)**i * phi * PL)
            else:
                U.append(0) # PL == 0 mean no update to U_L
        U_L = - np.sum(U)
    return U_L


def RFA_good_turing(t, prev, r, bootstrap=False):
    """
    estimate the number of unseen (r=1) species using the Rational Functions Approximation,
    see http://smithlabresearch.org/software/preseq/
    Can also estimate the number of species seen at least r-times (rSAC)

    This is a wrapper around the R library preseq, so R and that package need to be installed
    -> install.packages("preseqR")

    :param t: relative depth of the new sample compared to the old (i.e. t=2: the new sample has twice as many reads)
    :param prev: a pd.DataFrame with the CU information


    """

    # rpy2 is picky about data types
    assert isinstance(prev, pd.DataFrame)
    assert isinstance(t, float)
    assert isinstance(r, int)

    rpy2.robjects.r('library(preseqR)')

    if bootstrap:
        RFA_factory = rpy2.robjects.r('ds.rSAC.bootstrap')
    else:
        RFA_factory = rpy2.robjects.r('ds.rSAC')
    # covert dataframe to R
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(prev)

    fn = RFA_factory(r_from_pd_df, r)

    if bootstrap:
        f = fn[0](t)[0]  # the estimate
        se = fn[1](t)[0]  # the standard error
        lb = fn[2](t)[0]  # the lower confidence
        ub = fn[3](t)[0]  # the upper confidence
        res = {'f': f, 'se': se, 'lb': lb, 'ub': ub}
    else:
        res = {'f': fn(t)[0]}  # R returns vectors always, convert to number

    return res


def get_goodturing_df(CU_aggr: dict, tmax, gridpoints=500):
    """
    convenience function to do multiple estimators at once. Unfortunattely they differ slightly in how they define t.
    We standardize that here

    :param CU_aggr: a dictionary with the prevalences (freq of copy per molecule)
    :param tmax: maximum t (t-fold the original sample size) to estimate teh unseen species. If we set t=1 the estimator would be identity, t=2 asks for the number of species in a smaple of twice the size
    :param gridpoints: number of points in [1,t] to evaluate the estimator, default=500
    :returns: three dataframe, one per estimator. Contains the read-depth, the number of UMIs seen and the t-factor
    """

    # n_umi_Total = CU_aggr.get_nUMI() # sum(CU_aggr.values())
    # n_reads_Total = CU_aggr.get_nreads() # sum([k*v for k,v in CU_aggr.items() if k>0])

    n_umi_Total = sum(CU_aggr.values())
    n_reads_Total = sum([k*v for k, v in CU_aggr.items() if k > 0])

    t_GT = np.linspace(0.1, 1, 10) # the original GT estimator only works up to twice the sample size
    GT = pd.DataFrame([{'n_reads':n_reads_Total+n_reads_Total*t, 'n_umi': n_umi_Total+good_turing(t=t, prev=CU_aggr), 't': t} for t in t_GT])
    GT['estimator'] = 'GT'
    # for SGT, t is defined as: take a sample of t-times the original size. The estimator is the number of newly found species
    # its like generating a compmound sample old+new and ask for the total number of species observed
    t_SGT = np.linspace(1.01, tmax-1, gridpoints)  # goes from 2fold to x-fold
    SGT = pd.DataFrame([{
        'n_reads':n_reads_Total+n_reads_Total*t, 
        'n_umi': n_umi_Total+smoothed_good_turing(t=t, prev=CU_aggr), 
        't': t,
        } for t in t_SGT])
    SGT['n_umi_new'] = SGT['n_umi'] - n_umi_Total
    SGT['estimator'] = 'SGT'
    # for RFA thigns are defined a little different: given an old sample of size, if we take a new sample of size t x old, how many species do we see in the new sample (forgetting about the old)
    x = pd.DataFrame([{'i': k, 'n_i': v} for k,v in CU_aggr.items()]).sort_values('i')

    t_RFA = np.linspace(0.1, tmax, gridpoints)
    RFA = []
    for t in t_RFA:
        res = RFA_good_turing(t=float(t), prev=x, r=1, bootstrap=False)  # this also return confidence intervals
        res['n_umi'] = res['f']  # renaming
        del res['f']
        res['n_reads'] = n_reads_Total*t
        res['t'] = t
        res['n_umi_new'] = res['n_umi'] - n_umi_Total
        RFA.append(res)

    RFA = pd.DataFrame(RFA)
    RFA['estimator'] = 'RFA'

    return SGT, GT, RFA


def get_goodturing_df_nreads(CU_aggr, max_nreads):
    """
    the estimators usually work in terms of t, the t-fold size of the initial sample.
    Sometimes its more conveient to do it in terms of absolut reads though
    """
    n_reads_Total = CU_aggr.get_nreads()
    tmax = max_nreads / n_reads_Total

    return get_goodturing_df(CU_aggr, tmax)


def good_turing_R(A):
    """
    for a count vector A, this estimates the proportions, taking into account
    unseen species (which will make the proporrtions slightly smaller than the usual n_i/N)
    """
    import rpy2.robjects
    rpy2.robjects.r('library(edgeR)')
    goodTuringProportions = rpy2.robjects.r('goodTuringProportions')
    res = rpy2.robjects.IntVector(A)
    r = goodTuringProportions(res)
    p_g = np.array(r).flatten()
