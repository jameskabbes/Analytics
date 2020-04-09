import plotly as pl
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

kwargs_to_remove = ['example','show_plot']


class Fig:

    def __init__(self, subplot = False, **kwargs):

        if subplot:
            self.fig = make_subplots(**kwargs)
        else:
            self.fig = pgo.Figure(**kwargs)

    def add_trace(self, new_trace, **kwargs):
        self.fig.add_trace(new_trace, **kwargs)
        print (new_trace)

    def update_layout(self, **kwargs):
        self.fig.update_layout(**kwargs)

    def show(self, **kwargs):
        self.fig.show(**kwargs)

    def save(self, filename = 'image.png', **kwargs):
        print ('saving file ' + str(filename))
        self.fig.write_image(filename, **kwargs)

    def export_dict(self):
        return self.fig.to_dict()


class Axis:

    def __init__(self, example = False, **kwargs):

        if example:
            self.template = dict(showgrid = False, zeroline = False, nticks = 20, showline = True, title = 'X AXIS', mirror = 'all')
        else:
            self.template = dict( **kwargs )

    def add(self, **kwargs):
        for i in kwargs:
            self.template[i] = kwargs[i]

    def return_temp(self):
        return self.template









print ()
