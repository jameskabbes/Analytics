import plotly as pl
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import random


class Fig:

    def __init__(self, subplot = False, **kwargs):

        if subplot:
            self.fig = make_subplots(**kwargs)
        else:
            self.fig = pgo.Figure(**kwargs)

    def add_trace(self, new_trace, **kwargs):
        self.fig.add_trace(new_trace, **kwargs)

    def update_traces(self, **kwargs):
        self.fig.update_traces(**kwargs)

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
            self.template = dict(showgrid = False, zeroline = False, nticks = 20, showline = True, title = 'X AXIS', mirror = 'all', titlefont = Font(example = True).t)
        else:
            self.template = dict( **kwargs )

        self.t = self.template

    def add(self, **kwargs):
        for i in kwargs:
            self.template[i] = kwargs[i]

    def return_temp(self):
        return self.template

class Color:

    def __init__(self, type, tuple, example = False, **kwargs):

        self.type = type

        if tuple == None:
            self.gen_random()

        else:
            self.color = tuple

        self.error_check()
        self.get_string()

    def error_check(self):

        supported = ['rgb', 'rgba', 'cmyk']
        lens = [3, 4, 4]
        mins = [ (0,0,0), (0,0,0,0), (0,0,0,0)  ]
        maxs = [ (255,255,255), (255,255,255,1), (100,100,100,100) ]

        if self.type in supported:

            ind = supported.index(self.type)
            self.n = lens[ind]

            if len( self.color ) !=  lens[ind]:
                print ('Invalid color ' + str(self.color) + ' for type ' + str(self.type))
                print ('Expected length: ' + str(self.n))
                exit()

            for i in range(self.n):
                if self.color[i] > maxs[ind][i] or self.color[i] < mins[ind][i]:
                    print ('Invalid color ' + str(self.color) + ' for type ' + str(self.type))
                    print ('out of range for max and mins')
                    print ('MAX VALUES -> ' + str(maxs[ind]))
                    print ('MIN VALUES -> ' + str(maxs[ind]))
                    exit()


        else:
            print ('Color type ' + str(self.type) + ' not supported')
            exit()

    def get_string(self):

        string = self.type
        string += '('
        for i in self.color:
            string += ( str(i) + ',')
        string = string[:-1]
        string += ')'
        self.string = string

    def gen_random(self):

        if self.type == 'rgb':
            choices = list(range(256)) # 0 > 255
            color = []
            for i in range(3):
                color.append( random.choice(choices) )

        if self.type == 'rgba':
            self.type = 'rgb'
            self.gen_random()

            self.type = 'rgba'
            rgb = self.color
            rgb.append( random.random() )
            color = rgb

        if self.type == 'cmyk':
            chocies = list(range(0, 101))
            color = []
            for i in range(4):
                color.append( random.choice(choices))

        print (self.type)


        self.color = color

class Font:

    def __init__(self, example = False, **kwargs):

        if example:
            self.template = dict(family = 'Arial', size = 16)
        else:
            self.template = dict( **kwargs )

        self.t = self.template

    def add(self, **kwargs):
        for i in kwargs:
            self.template[i] = kwargs[i]

    def return_temp(self):
        return self.template


def get_colorway(type = 'rgb', n = 1, color_classes = []):

    if color_classes == []:
        for i in range(n):
            #get random
            color_classes.append( Color(type, None))

    strings = []
    for color in color_classes:
        strings.append( color.string )

    return strings

def get_cont_color_range( color_classes, type = 'rgb', props = [] ):

    if props == []:
        props = np.linspace(0, 1, len(color_classes))
    color_range = []

    for i in range(len(color_classes)):
        color = color_classes[i]
        string = color.string
        color_range.append( (props[i], string)  )

    return color_range

def gen_layout(example = True, **kwargs):

    comb_args = {}

    if example:
        add_args = dict(
        height = 800, width = 1200, title = 'A Sweet Graph',
        xaxis = Axis(title = 'X AXIS').template,
        yaxis = Axis(title = 'Y AXIS').template,
        colorway = get_colorway( 'rgb', color_classes = [Color('rgb', None)] )
        )
        comb_args.update(add_args)

    comb_args.update(**kwargs)
    return pgo.Layout(**comb_args)

def plot(type, example = False, show_plot = False, **kwargs):

    types = ['bar','scatter','line','heatmap','histogram','box','scattergeo']
    funcs = [ bar,   scatter,  line,  heatmap,  histogram,  box,  scattergeo ]

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
        colorscale = [ (0, 'rgb(255,255,255)'), (1, 'rgb(255,0,0)')]
        add_args = dict( z = z, colorscale = colorscale )

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

        fig = Fig( data = plot(type, example = True), layout = gen_layout() )
        fig.show()
