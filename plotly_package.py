import plotly as pl
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

kwargs_to_remove = ['example','show_plot']


class Fig:

    def __init__(self, **kwargs):
        self.fig = pgo.Figure(**kwargs)
        self.data = []
        self.layout = []

        if 'data' in kwargs:
            self.data = kwargs['data']
        if 'layout' in kwargs:
            self.layout = kwargs['data']

    def new_data(self, new_data):
        self.data = new_data

    def add_trace(self, new_trace):
        self.data.append(new_trace)

    def gen_layout(self, **kwargs, update = True):
        layout = pgo.Layout(**kwargs)
        if update:
            self.update_layout(layout)

    def update_layout(self, layout):
        self.layout = layout

    def show(self):
        self.fig.show()

    def export(self, filename):
        write_image(self.fig, filename)


def set_axis_layout(**axis_kwargs):

    def example():
        template = dict(showgrid = False, zeroline = False, nticks = 20, showline = True, title = 'X AXIS', mirror = 'all')
        return template

    template = check_example(example, **axis_kwargs)
    axis_kwargs, non_axis_kwargs = sep_kwargs(**axis_kwargs)

    if template == None:
        template = dict (**axis_kwargs)

    return template

def set_layout(**layout_kwargs):

    def example():
        xaxis = set_axis_layout(title = 'X Axis')
        yaxis = set_axis_layout(title = 'Y Axis')
        layout = pgo.Layout(xaxis = xaxis, yaxis = yaxis, title = 'Example Title')
        return layout

    layout = check_example(example, **layout_kwargs)
    layout_kwargs, non_layout_kwargs = sep_kwargs(**layout_kwargs)

    if layout == None:
        layout = pgo.Layout(**layout_kwargs)

    return layout

def show_fig(data, **plot_kwargs):

    layout = set_layout(**plot_kwargs)
    fig = pgo.Figure(data = data, layout = layout)
    fig.show()
    write_image(fig, 'asdf.png')

def write_image(fig, filename):

    fig.write_image(filename)

def make_subplot_fig(**plot_kwargs):

    fig = make_subplots(**plot_kwargs)
    return fig

def add_to_subplot_fig(fig, trace, **kwargs):

    fig.add_trace(trace, **kwargs)
    return fig

def update_layout_subplot(fig, **kwargs):

    fig.update_layout(**kwargs)
    return fig

def add_xaxis_subplot(fig, **kwargs):

    fig.update_xaxes(**kwargs)
    return fig

def add_yaxis_subplot(fig, **kwargs):

    fig.update_yaxes(**kwargs)
    return fig

def add_all_to_subplot(data, data_kwargs, layout_kwargs, x_axis_kwargs, y_axis_kwargs, **plot_kwargs):

    def example():

        x = np.linspace(1, 6, 6)
        x_noise = x + (np.random.rand(6) * .1)
        trace1 = scatter(x, x**2, name = 'Trace 1')
        trace2 = scatter(x, 10*x_noise**2, name = 'Trace 2')
        trace3 = scatter(x, x**1.5, name = 'Trace 3')
        trace4 = scatter(x, 10*x_noise**1.5, name = 'Trace 4')

        data = [trace1, trace2, trace3, trace4]
        data_kwargs = [ dict( secondary_y = False, row = 1, col = 1), dict( secondary_y = True, row = 1, col = 1),
        dict( secondary_y = False, row = 1, col = 2), dict( secondary_y = True, row = 1, col = 2) ]

        xaxis = [ {'title_text': 'this will not appear', 'row': 1, 'col': 1},{'title_text': 'Title 1 x axis', 'row': 1, 'col' : 1},{'title_text': 'this will not appear', 'row': 1, 'col' : 2},{'title_text': 'Title 2 x axis', 'row': 1, 'col' : 2}  ]
        yaxis = [ {'title_text': 'Secondary Axis 1', 'row': 1, 'col': 1, 'secondary_y': True},{'title_text': 'Y axis 1', 'row': 1, 'col' : 1, 'secondary_y' : False},{'title_text': 'Secondary Axis 2', 'row': 1, 'col' : 2, 'secondary_y': True},{'title_text': 'Y axis 2', 'row': 1, 'col' : 2, 'secondary_y': False} ]

        layout_kwargs = dict( title = '1x2 Subplot' )
        plot_kwargs = dict( rows = 1, cols = 2, specs = [  [{'secondary_y': True}, {'secondary_y': True}]  ] )

        return data, data_kwargs, layout_kwargs, xaxis, yaxis, plot_kwargs

    if 'example' in plot_kwargs:
        if plot_kwargs['example']:
            data, data_kwargs, layout_kwargs, x_axis_kwargs, y_axis_kwargs, plot_kwargs = example()
            plot_kwargs['example'] = 'filler val'

        del plot_kwargs['example']

    fig = make_subplot_fig(**plot_kwargs)

    if type(data[0]) == list:
        nested = True
    else:
        nested = False

    if nested:
        rows = plot_kwargs['rows']
        cols = plot_kwargs['cols']
        for i in range(rows):
            for j in range(cols):

                kwargs = data_kwargs[i][j]
                trace = data[i][j]

                if 'row' not in kwargs:
                    kwargs.update( dict(row = i + 1))
                if 'col' not in kwargs:
                    kwargs.update( dict(col = j + 1))

                print (i,j)
                print (x_axis_kwargs[i][j])
                print (y_axis_kwargs[i][j])


                fig = add_to_subplot_fig(fig, trace, **kwargs)
                fig = add_xaxis_subplot(fig, **x_axis_kwargs[i][j], row = i+1, col = j+1)
                fig = add_yaxis_subplot(fig, **y_axis_kwargs[i][j], row = i+1, col = j+1)

    else:
        for i in range(len(data)):
            kwargs = data_kwargs[i]
            trace = data[i]

            print (i)
            print (x_axis_kwargs[i])
            print (y_axis_kwargs[i])

            fig = add_to_subplot_fig(fig, trace, **kwargs)
            fig = add_xaxis_subplot(fig, **x_axis_kwargs[i])
            fig = add_yaxis_subplot(fig, **y_axis_kwargs[i])

    fig = update_layout_subplot(fig, **layout_kwargs)
    fig.show()
    return fig


def sep_kwargs(**kwargs):

    '''separates the keyword arguments into lists of arguments used for plotting
    and those used for the functions'''

    keep_kwargs = {}
    remove_kwargs = {}

    for kwarg in kwargs:
        if kwarg in kwargs_to_remove:
            remove_kwargs.update({kwarg: kwargs[kwarg]})
        else:
            keep_kwargs.update({kwarg: kwargs[kwarg]})

    return keep_kwargs, remove_kwargs

def check_example(example_func, **kwargs):

    '''Returns example data if specified in kwargs, returns kwargs without the 'example' parameter'''

    if 'example' in kwargs:

        val = kwargs['example']
        if val:
            data = example_func()
            return data
        else:
            return None
    else:
        return None

def bar(values, **kwargs):

    def example():
        x = np.linspace(0,9,10)
        y = x ** 2
        data = pgo.Bar(x = x, y = y)
        return data

    data= check_example(example, **kwargs)
    plot_kwargs, non_plot_kwargs = sep_kwargs(**kwargs)

    if data == None:
        data = pgo.Bar(y = values, **plot_kwargs)

    if 'show_plot' in kwargs:
        if kwargs['show_plot']:
            show_fig(data)

    return data

def scatter(x, y, **kwargs):

    def example():
        x = np.linspace(0, 9, 10)
        y = x ** 2
        data = pgo.Scatter(x = x, y = y)
        return data

    data = check_example(example, **kwargs)
    plot_kwargs, non_plot_kwargs = sep_kwargs(**kwargs)

    if data == None:
        data = pgo.Scatter(x = x, y = y, **plot_kwargs)

    if 'show_plot' in kwargs:
        if kwargs['show_plot']:
            show_fig(data)

    return data

def line(x, y, **kwargs):

    def example():
        x = np.linspace(0, 9, 10)
        y = x ** 2
        data = pgo.Line(x = x, y = y)
        return data

    data = check_example(example, **kwargs)
    plot_kwargs, non_plot_kwargs = sep_kwargs(**kwargs)

    if data == None:
        data = pgo.Scatter(x = x, y = y, **kwargs)

    if 'show_plot' in kwargs:
        if kwargs['show_plot']:
            show_fig(data)

    return data

def heatmap(grid, **kwargs):

    def example():
        data = [
        [2,3,4,5],
        [5,6,7,8],
        [1,2,3,4] ]
        trace = pgo.Heatmap(z = data)
        return trace

    data = check_example(example, **kwargs)
    plot_kwargs, non_plot_kwargs = sep_kwargs(**kwargs)

    if data == None:
        data = pgo.Heatmap(z = data, **plot_kwargs)

    if 'show_plot' in kwargs:
        if kwargs['show_plot']:
            show_fig(data)

    return data

def histogram(values, **kwargs):

    def example():
        data = np.random.randn(1000)
        trace = pgo.Histogram(x = data)
        return trace

    data = check_example(example, **kwargs)
    plot_kwargs, non_plot_kwargs = sep_kwargs(**kwargs)

    if data == None:
        data = pgo.Heatmap(x = data, **plot_kwargs)

    if 'show_plot' in kwargs:
        if kwargs['show_plot']:
            show_fig(data)

    return data

def box(values, **kwargs):

    def example():
        data = np.random.randn(1000)
        trace = pgo.Box(x = data)
        return trace

    data = check_example(example, **kwargs)
    plot_kwargs, non_plot_kwargs = sep_kwargs(**kwargs)

    if data == None:
        data = pgo.Heatmap(x = data, **plot_kwargs)

    if 'show_plot' in kwargs:
        if kwargs['show_plot']:
            show_fig(data)

    return data

def scattergeo(lon, lat, **kwargs):

    def example():

        df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
        df['text'] = df['airport'] + '' + df['city'] + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)

        trace = pgo.Scattergeo(
        lon = df['long'],
        lat = df['lat'],
        text = df['text'],
        mode = 'markers',
        marker_color = df['cnt'])

        print ('overriding and plotting')
        show_fig([trace], title = 'Most trafficked US airports<br>(Hover for airport names)', geo_scope='usa')

        return trace

    data = check_example(example, **kwargs)
    plot_kwargs, non_plot_kwargs = sep_kwargs(**kwargs)

    if data == None:
        data = pgo.Scattergeo(lon = lon, lat = lat, **plot_kwargs)

    if 'show_plot' in kwargs:
        if kwargs['show_plot']:
            show_fig(data)

    return data



def plot(type, *args, **kwargs):

    types = ['bar','scatter','line','heatmap','histogram','box','scattergeo']
    funcs = [ bar,  scatter,  line,  heatmap,  histogram,  box,  scattergeo ]

    try:
        ind = types.index(type)
    except:
        print ('No known function')

    func = funcs[ind]

    trace = func(*args, **kwargs)
    return trace



def plot_mult_main(type, data_args, data_kwargs, **plotting_kwargs):

    def example():

        print ('plotting multiple')

        x = np.linspace(0, 9, 10)
        ys = [ x**1, x**1.5, x**2 ]

        type = 'bar'
        data_kwargs = [dict(x = x, name = 'Dataset 1', show_plot = False), dict(x = x, name = 'Dataset 2', show_plot = False), dict(x = x, name = 'Dataset 3', show_plot = False) ]

        trace1 = plot(type, ys[0], **data_kwargs[0])
        trace2 = plot(type, ys[1], **data_kwargs[1])
        trace3 = plot(type, ys[2], **data_kwargs[2])
        data = [trace1, trace2, trace3]

        xaxis = set_axis_layout(title = 'X AXIS')
        yaxis = set_axis_layout(title = 'Y AXIS')
        plotting_kwargs = dict(xaxis = xaxis, yaxis = yaxis, title = 'this is a test title')

        show_fig(data, **plotting_kwargs)

    to_plot = True
    if 'example' in plotting_kwargs:
        if plotting_kwargs['example']:
            example()
            to_plot = False
        del plotting_kwargs['example']

    if to_plot:

        for i in range(len(data_args)):

            args = data_args[i]
            kwargs = data_kwargs[i]

            kwargs.update( {'show_plot': False} )

            trace = plot(type, *args, **kwargs)
            data.append(trace)

        show_fig(data, **plotting_kwargs)


if __name__ == '__main__':

    ##mult axes

    '''
    x = np.linspace(1, 6, 6)
    x_noise = x + (np.random.rand(6) * .1)
    trace1 = scatter(x, x**2, name = 'Trace 1')
    trace2 = scatter(x, 10*x_noise**2, name = 'Trace 2')
    trace3 = scatter(x, x**1.5, name = 'Trace 3')
    trace4 = scatter(x, 10*x_noise**1.5, name = 'Trace 4')

    data = [ [trace1, trace2], [trace3, trace4] ]
    layout_kwargs = dict( title = '2x2 Subplot' )

    xaxis = [ [{'title_text': 'Title 1 x axis'},{'title_text': 'Title 2 x axis'}], [{'title_text': 'Title 3 x axis'},{'title_text': 'Title 4 x axis'}] ]
    yaxis = [ [{'title_text': 'Y axis 1'},{'title_text': 'Y axis 2'}], [{'title_text': 'Y axis 3'},{'title_text': 'Y axis 4'}] ]

    add_all_to_subplot( data, [ [{},{}], [{},{}] ] , layout_kwargs, xaxis, yaxis, rows = 2, cols = 2, subplot_titles = ('Plot 1','Plot 2','Plot 3','Plot 4'), example = False )
    '''

    bar([1,2,3,4], example = True, show_plot = True)







print ()
