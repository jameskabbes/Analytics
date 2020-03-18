import matplotlib.pyplot as plt
import numpy as np

def ramp(nested_values, dt = .1, legend = []):

    cutoffs = np.linspace(0, 100, 100 / dt)
    print (cutoffs)

    for i in nested_values:

        vals = np.array(i)
        vals = np.sort(vals)

        n = len(vals)

        x = []

        for prop in cutoffs:

            ind = int(prop * n / 100)
            filtered = vals[ : ind]

            try:
                x.append( filtered[-1] )
            except:
                x.append( vals[0])

        plt.plot(x, cutoffs)

    if legend == []:
        for i in range(len(nested_values)):
            legend.append( str(i))

    plt.legend(legend)
    plt.show()







if __name__ == '__main__':

    target = 1000

    #first group
    start = 0
    values1 = []

    value = start
    while value < target:

        values1.append(value)
        value += 1

    ##second gorup
    start = 0
    values2 = []

    adder = 0

    value = start
    while value < target:

        values2.append(value)
        value += (1+ adder)
        adder += .01

    ##second gorup
    start = 0
    values3 = []

    adder = 0

    value = start
    while value < target:

        values3.append(value)
        value += (1+ adder)
        adder -= .0005


    print (values1)
    print (values2)
    print (values3)


    ramp( [values1, values2, values3], legend = ['A','B','C'])


#exit
