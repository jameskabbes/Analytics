import plotly_package as plp
import pandas as pd
import numpy as np

def ramp(nested_values, example = False, show_plot = False, **kwargs):


    if example:
        nested_values = []
        x = np.linspace(1, 1000, 1000)
        nested_values.append( x ** 1 )
        nested_values.append( x ** 1.05  )
        nested_values.append( x ** 1.11 )

    data = []

    for i in nested_values:

        dt = 1 / len(i)
        cutoffs = np.linspace(0, 100, 100 / dt)
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

        trace = plp.line(x = x , y= cutoffs, **kwargs)
        data.append(trace)

    if show_plot:
        fig = plp.Fig(data = data)
        xaxis = plp.Axis(title = 'Values').template
        yaxis = plp.Axis(title = 'Percent of Inputs < X').template
        fig.update_layout(example = True, xaxis = xaxis, yaxis = yaxis)
        fig.show()

    return data
