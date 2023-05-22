import numpy as np

def cdf (ax, data, label):
    N=len(data)
    # sort the data in ascending order
    x = np.sort(data)
    # get the cdf values of y
    y = 1 - (np.arange(N) / float(N))
    # plotting
    ax.set_xlabel(label)
    ax.set_ylabel('CDF of %s' %label.lower())
    ax.plot(x, y, marker='o', markersize = 1.5, linewidth =0)
    #ax.title('cdf of %s in log log scale' %label)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return(ax)
    
def part2dict(A):
    """
    Given a partition (list of sets), returns a dictionary mapping the part for each vertex
    """
    x = []
    for i in range(len(A)):
        x.extend([(a, i) for a in A[i]])
    return {k: v for k, v in x}