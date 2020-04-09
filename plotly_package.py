import plotly as pl
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class Fig:

    def __init__(self, subplot = False, **kwargs):

        if subplot:
            self.fig = make_subplots(**kwargs)
        else:
            self.fig = pgo.Figure(**kwargs)

    def add_trace(self, new_trace, **kwargs):
        self.fig.add_trace(new_trace, **kwargs)
        print (new_trace)

    def update_layout(self, example = False, **kwargs):

        comb_args = {}

        if example:
            add_args = dict(
            height = 800, width = 1200, title = 'A Sweet Graph',
            xaxis = Axis(title = 'X AXIS').template,
            yaxis = Axis(title = 'Y AXIS').template,
            )
            comb_args.update(add_args)

        comb_args.update(**kwargs)
        self.fig.update_layout(**comb_args)

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


def gen_layout(example = True, **kwargs):

    comb_args = {}

    if example:
        add_args = dict(
        height = 800, width = 1200, title = 'A Sweet Graph',
        xaxis = Axis(title = 'X AXIS').template,
        yaxis = Axis(title = 'Y AXIS').template,
        )
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    return pgo.Layout(**comb_args)

def plot(type, example = False, show_plot = False, **kwargs):

    types = ['bar',    'scatter',    'line',    'heatmap',    'histogram',    'box',    'scattergeo']
    funcs = [ bar,  scatter,  line,  heatmap,  histogram,  box,  scattergeo ]

    try:
        ind = types.index(type)
        func = funcs[ind]
    except:
        print ('No known function -> trying type as function input')
        func = type

    trace = func(example = example, show_plot = False, **kwargs)

    if show_plot:
        fig = Fig(data = trace)
        fig.show()

    return trace

def bar(example = False, show_plot = False, **kwargs):

    comb_args = {}

    if example:
        x = np.linspace(0,9,10)
        y = x ** 2
        add_args = dict( x = x,  y= y)
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    trace = pgo.Bar(**comb_args)

    if show_plot:
        fig = Fig(data =trace)
        fig.show()

    return trace

def scatter(example = False, show_plot = False, **kwargs):

    comb_args = {}

    if example:
        x = np.linspace(0,9,10)
        y = x ** 2
        add_args = dict( x = x,  y= y)
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    trace = pgo.Scatter(**comb_args)

    if show_plot:
        fig = Fig(data =trace)
        fig.show()

    return trace

def line(example = False, show_plot = False, **kwargs):

    comb_args = {}

    if example:
        x = np.linspace(0, 9, 10)
        y = x ** 2
        add_args = dict( x = x,  y= y)
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    trace = pgo.Line(**comb_args)

    if show_plot:
        fig = Fig(data =trace)
        fig.show()

    return trace

def heatmap(example = False, show_plot = False, **kwargs):

    comb_args = {}

    if example:
        z = [
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6],
        [4,5,6,7]]
        add_args = dict( z = z )
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    trace = pgo.Heatmap(**comb_args)

    if show_plot:
        fig = Fig(data =trace)
        fig.show()

    return trace

def histogram(example = False, show_plot = False, **kwargs):

    comb_args = {}

    if example:
        data = np.random.randn(1000)
        add_args = dict(x = data)
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    trace = pgo.Histogram(**comb_args)

    if show_plot:
        fig = Fig(data = trace)
        fig.show()

    return trace

def box(example = False, show_plot = False, **kwargs):

    comb_args = {}

    if example:
        data = np.random.randn(1000)
        add_args = dict( x = data )
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    trace = pgo.Box(**comb_args)

    if show_plot:
        fig = Fig(data =trace)
        fig.show()

    return trace

def scattergeo(example = False, show_plot = False, **kwargs):

    comb_args = {}

    if example:
        df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
        df['text'] = df['airport'] + '' + df['city'] + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)

        add_args = dict( lon = df['long'],
        lat = df['lat'],
        text = df['text'],
        mode = 'markers',
        marker_color = df['cnt'] )
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    trace = pgo.Scattergeo(**comb_args)

    if show_plot:
        fig = Fig(data = trace)
        if example:
            fig.update_layout(title = 'Most trafficked US airports<br>(Hover for airport names)', geo_scope='usa' )

        fig.show()

    return trace


if __name__ == '__main__':

    types = ['bar','scatter','line','heatmap','histogram','box','scattergeo']
    for type in types:
        plot(type, example = True, show_plot = True)
