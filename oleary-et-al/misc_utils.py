import matplotlib.pyplot as plt

'''
    Plotting functions
'''

def xys_plot(plt, xss, yss, xlabel, ylabel, legends, markers, title, alphas, legend=True):
    for i in range(len(xss)):
        plt.plot(xss[i], yss[i], markers[i], alpha=alphas[i], label=legends[i])
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend:
        plt.legend(loc='best', prop={'size': 10})

    plt.grid()
    plt.show()


def dict_sum(a, b):
    '''Takes 2 dictionaries, assumed to contain the same type of values, and sums the values.'''
    for key in b:
        if key in a:
            a[key] += b[key]
        else:
            a[key] = b[key]
    
    return a


