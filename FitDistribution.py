
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import warnings
import numpy as np
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def best_fit_distribution(data, bins=200, ax=None):
    print("VASILIS _distn_names: ", _distn_names)

    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))  
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def make_cdf(dist, params, size=10000):
    """Generate distributions's Cumulative Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    value = 0.017089
    cdf_prob = dist.cdf(value,*arg, loc=loc,scale=scale); 
    print("cdf:", cdf_prob)
    inverse_cdf_prob = dist.ppf(cdf_prob, *arg, loc=loc, scale=scale)
    print("Inverse cdf:", inverse_cdf_prob)

    ppf80 = dist.ppf(0.8, *arg, loc=loc, scale=scale)
    print("ppf 80%:", ppf80)

    ppf20=dist.ppf(0.2, *arg, loc=loc, scale=scale)
    print("ppf 20%:", ppf20)

    # Get sane start and end points of distribution
    #start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    #end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    #x = np.linspace(start, end, size)
    #y = dist.cdf(x, loc=loc, scale=scale, *arg)
    #cdf = pd.Series(y, x)

    return cdf_prob

def getBestFitDistributions(data):
    # Find best fit distribution
    #best_distibutions = best_fit_distribution(data, 200)

        # Plot for comparison
    plt.figure(figsize=(12,8))
    #ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])
    ax = plt.hist(data, bins=50, density=True, label='Data')
    #plt.title("Distance Error (Orig - Approx) - Distribution")
    #plt.show()

    # Save plot limits
#    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_distibutions = best_fit_distribution(data, 200, ax)
    best_dist = best_distibutions[0]
    print("VASILIS best_dist: ", best_dist)

    # Update plots
 #   ax.set_ylim(dataYLim)
    plt.title(u'All Fitted Distributions')
    plt.xlabel(u'Original-Approx Distance Error Metric')
    plt.ylabel('Frequency')

    # Make PDF with best params 
    pdf = make_pdf(best_dist[0], best_dist[1])

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    #data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
    plt.hist(data, bins=50, density=True, label='Data')

    param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
    dist_str = '{}({})'.format(best_dist[0].name, param_str)

    ax.set_title(u'Best Fitted Distribution\n' + dist_str)
    ax.set_xlabel(u'Original-Approx Distance Error Metric')
    ax.set_ylabel('Frequency')

    plt.show()

    cdf = make_cdf(best_dist[0], best_dist[1])
    print("1 - CDF: ", 1 - cdf)

    return best_distibutions

def getBestFitDistribution(data):
     # Find best fit distribution
    best_distibutions = best_fit_distribution(data, 200)
    best_dist = best_distibutions[0]
    print("VASILIS best_dist: ", best_dist)
    return best_dist


