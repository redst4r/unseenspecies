import numpy as np
from scipy.stats import binom
import pandas as pd
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.robjects


def custom_log(x, base):
    return np.log(x) / np.log(base)


def good_turing(t, prev):
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


def smoothed_good_turing(t, prev):
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


def RFA_good_turing(t, prev, r):
    """
    estimate the number of unseen (r=1) species using the Rational Functions Approximation,
    see http://smithlabresearch.org/software/preseq/
    Can also estimate the number of species seen at least r-times (rSAC)

    This is a wrapper around the R library preseq, so R and that package need to be installed

    :param t: relative depth of the new sample compared to the old (i.e. t=2: the new sample has twice as many reads)
    :param prev: a pd.DataFrame with the CU information


    """

    # rpy2 is picky about data types
    assert isinstance(prev, pd.DataFrame)
    assert isinstance(t, float)
    assert isinstance(r, int)

    rpy2.robjects.r('library(preseqR)')
    RFA_factory = rpy2.robjects.r('ds.rSAC')
    # covert dataframe to R
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(prev)

    fn = RFA_factory(r_from_pd_df, r)

    res = fn(t)[0]  # R returns vectors always, convert to number
    return res


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
