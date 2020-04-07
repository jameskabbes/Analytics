import plotly as pl
import plotly.graph_objects as pgo
import pandas as pd
import numpy as np

kwargs_to_remove = ['example','show_plot']

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

    layout = pgo.Layout(**layout_kwargs)
    return layout

def show_fig(data, **plot_kwargs):

    layout = set_layout(**plot_kwargs)
    fig = pgo.Figure(data = data, layout = layout)
    fig.show()

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
        data = pgo.Bar(x = x, y = y)
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

    if type == 'bar':
        func = bar

    elif type == 'scatter':
        func = scatter

    elif type == 'line':
        func = line

    elif type == 'heatmap':
        func = heatmap

    elif type == 'histogram':
        func = histogram

    elif type == 'box':
        func = box

    elif type == 'scattergeo':
        func = scattergeo

    else:
        print ('Type ' + str(type) + ' unknown')

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

    trace = plot('scattergeo', [], [], example = True)
    show_fig(trace)
















print ()
