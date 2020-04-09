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


def plot(type, example = False, show_plot = False, **kwargs):

    types = ['bar',    'scatter',    'line',    'heatmap',    'histogram',    'box',    'scattergeo',    'ramp']
    funcs = [ bar,  scatter,  line,  heatmap,  histogram,  box,  scattergeo,  ramp ]

    try:
        ind = types.index(type)
        func = funcs[ind]
    except:
        print ('No known function -> trying type as function input')
        func = type

    trace = func(example = example, show_plot = False, **kwargs)

    if show_plot:
        fig = Fig(data = trace)
        fig.show_fig()

    return trace

def bar(example = False, show_plot = False, **kwargs):

    if example:
        x = np.linspace(0,9,10)
        y = x ** 2
        add_args = dict( x = x,  y= y)
        kwargs.update(add_args)

    trace = pgo.Bar(**kwargs)

    if show_plot:
        fig = Fig(data =trace)
        fig.show_fig()

    return trace

def scatter(example = False, show_plot = False, **kwargs):

    if example:
        x = np.linspace(0,9,10)
        y = x ** 2
        add_args = dict( x = x,  y= y)
        kwargs.update(add_args)

    trace = pgo.Scatter(**kwargs)

    if show_plot:
        fig = Fig(data =trace)
        fig.show_fig()

    return trace

def line(example = False, show_plot = False, **kwargs):

    if example:
        x = np.linspace(0, 9, 10)
        y = x ** 2
        add_args = dict( x = x,  y= y)
        kwargs.update(add_args)

    trace = pgo.Line(**kwargs)

    if show_plot:
        fig = Fig(data =trace)
        fig.show_fig()

    return trace

def heatmap(example = False, show_plot = False, **kwargs):

    if example:
        z = [
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6],
        [4,5,6,7]]
        add_args = dict( z = z )
        kwargs.update(add_args)

    trace = pgo.Line(**kwargs)

    if show_plot:
        fig = Fig(data =trace)
        fig.show_fig()

    return trace

def box(example = False, show_plot = False, **kwargs):

    if example:
        data = np.random.randn(1000)
        add_args = dict( x = data )
        kwargs.update(add_args)

    trace = pgo.Box(**kwargs)

    if show_plot:
        fig = Fig(data =trace)
        fig.show_fig()

    return trace

def scattergeo(example = False, show_plot = False, **kwargs):

    if example:
        df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
        df['text'] = df['airport'] + '' + df['city'] + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)

        add_args = dict( lon = df['long'],
        lat = df['lat'],
        text = df['text'],
        mode = 'markers',
        marker_color = df['cnt'] )

        kwargs.update(add_args)

    trace = pgo.Scattergeo(**kwargs)

    if show_plot:
        fig = Fig(data = trace)
        if example:
            fig.update_layout(title = 'Most trafficked US airports<br>(Hover for airport names)', geo_scope='usa' )

        fig.show()

    return trace
