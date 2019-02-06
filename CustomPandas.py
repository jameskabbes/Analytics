#DataFrame functions

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#only import if ya need plots

import collections
#sorted

import numpy as np

def get_df(file_name, delim = ','):
    '''Read csv file from local directory: return dataframe'''

    df = pd.read_csv(file_name, header = 0,  sep = delim)
    return df

def replace_in_df(dataframe, column, to_find, to_replace):

    '''Replaces list or string "to_find" with list or string "to_replace"
    Looks up all instances in "column" found in the dataframe and replaces them
    Returns dataframe'''

    assert type(column) == str

    if type(to_find) == str:
        to_find = [to_find]

    if type(to_find) == list:

        if type(to_replace) == list:
            assert len(to_replace) == len(to_find) #replace them based on index match
            # to_find = ['USA','GB'] to_replace = ['United States','Great Britain']

        else:
            #in this case all the instances in to_find will be replaced with the string to_replace
            a = to_replace
            to_replace = [a, ] * len(to_find)

        for i in range(len(to_find)):
            item = to_find[i]
            replace = to_replace[i]
            dataframe.loc[ dataframe[column] == item, column  ] = replace

    return dataframe

def keep_these_cols(df, cols):
    '''Returns dataframe with columns 'cols' '''
    return df[cols]

def drop_these_cols(df, cols):
    '''Returns dataframe without columns 'cols' '''
    return df.drop([cols], axis = 1)

def keep_these_rows(df, rows):
    '''Returns dataframe with index values contained in 'rows' '''
    return df.loc(rows)

def drop_these_rows(df, rows):
    '''Returns dataframe without 'rows' based on index value'''
    return df.drop(rows)

def sort_df(df, columns, ascend, na_pos = 'last', reset_ind = True):

    '''Sorts a dataframe based on columns and a boolean list 'ascend' on whether should a ascend or descend' '''
    ###Permanently sort

    assert len(columns) == len(ascend)
    #sorts by first column first, then by second column

    df.sort_values(columns, ascending = ascend, na_position = na_pos)
    #df = df.sort_values(['Age','Publications'], ascending = [1,1], na_position = na_pos)

    #print (df)
    if reset_ind:
        df = df.reset_index()
    return df

def assign_column_based_on_existing(df, columns, scenarios):

    ###Needs work to handle x number of dimensions
    # how to utilize vector optimization for x-number of dimensions without hard coding???

    '''Assigns values to columns[-1] based on what the values in the previous columns stated
    columns = ['Age','Gender', 'New column']
    scenarios = [ [21, 'Male', '21-year-old male'], [21, 'Female', '21-year-old female'] ]'''
    #df is a pandas df
    #columns is a list of strings
    #scenarios is nested list!  n number of lists with each list (len(columns)) long

    #when df[col1] == scen1 and df[col2] == scen2 set df[col3] == scen3
    assert type(scenarios[0]) == list
    assert len(columns) == len(scenarios[0])
    assert len(columns) >= 2
    #The last column is the one you write to


    #Is there a better way to write this code in general with any dimension??
    for scenario in scenarios:


        if len(columns) == 2:
            a, write = scenario
            df.loc[ df[ columns[0] ] == a  ,  columns[-1] ] = write

        elif len(columns) == 3:
            a, b,write = scenario
            df.loc[ ( (df[columns[0]] == a) & (df[columns[1]] == b) )  ,  columns[-1] ] = write

        elif len(columns) == 4:
            a, b,c, write = scenario
            df.loc[ ( (df[columns[0]] == a) & (df[columns[1]] == b) & (df[columns[2]] == c) )  ,  columns[-1] ] = write

        elif len(columns) == 5:
            a, b,c,d,write = scenario
            df.loc[ ( (df[columns[0]] == a) & (df[columns[1]] == b) & (df[columns[2]] == c) & (df[columns[3]] == d))  ,  columns[-1] ] = write

        else:
            print ('not designed this many columns...add code to the function')

    return df

def rename_cols(df, existing, new):

    '''renames cols found in 'existing' to match those found in "new" '''

    if type(existing) == str:
        existing = [existing]
    if type(new) == str:
        new = [new]

    assert len(existing) == len(new)

    for i in range(len(existing)):
        df = df.rename(index = str, columns = {existing[i] : new[i]})

    return df

def ceiling_filter(df, max_val, column):
    '''Returns df with values below 'max_val' found in 'column' '''
    df = df[  df[column] <= max_val  ]
    return df

def floor_filter(df, min_val, column):
    '''Returns df with values above 'min_val' found in 'column' '''
    df = df[  df[column] >= min_val  ]
    return df

def boxplot(df, column_with_values, group_by = None):
    '''Shows a boxplot values found in 'column_with_values' with the option to group by another column'''

    if type(group_by) == str:
        df.boxplot(column = column_with_values, by = group_by)
    else:
        df.boxplot(column = column_with_values)

def add_to_bottom(df, column_names, values):

    '''Adds to bottom of dataframe based on 'column_names' and 2D list 'values.' See append_to_df_with_df for appending dfs'''

    assert len(values) == len(column_names)

    if type(column_names) == str:
        column_names = [column_names]
    if type(values) == str:
        values = [values]

    dictionary = {}
    for i in range(len(values)):
        dictionary[column_names[i]] = values[i]

    df = append_to_df_with_df(df, dict_to_df(dictionary))
    return df

def list_of_index_values(df):

    '''Return a list of all the index values found in df'''
    return df.index.tolist()

def append_to_df_with_df(og_df, new_df, reset_ind = True):
    '''Appends 'og_df' with 'new_df' and returns new'''
    return og_df.append(new_df, ignore_index = reset_ind)

def grab_rows_with_certain_values(df, column, values, return_not_in_values = False):

    '''Returns a version of the dataframe where every row in "column" contains a value found in list"values"'''

    if return_not_in_values: #this means we want only the values that are not contained in the values list

        if type(values) == list:
            #print ( df.loc[~df[column].isin(values)] )
            df = df.loc[~df[column].isin(values)]
        else:
            df = df.loc[df[column] != values]

    else: #only those not in the values list

        if type(values) == list:
            #print ( df.loc[df[column].isin(values)] )
            df = df.loc[df[column].isin(values)]
        else:
            df = df.loc[df[column] == values]


    return df

def violin_plot(df, column, filter_column = None, min_val = None, max_val = None, mult = 1, add = 0):

    '''Displays a violin plot of the values found in "column" filtered by optional param "filter_column"'''

    if min_val != None:
        df = floor_filter(df, min_val, column)
    if max_val != None:
        df = ceiling_filter(df, max_val, column)

    df[column] = df[column] * mult
    df[column] = df[column] + add
    if filter_column != None:
        a = sns.violinplot(x = df[filter_column], y = df[column], inner = 'quartiles')
    else:
        a = sns.violinplot(y = df[column], inner = 'quartiles')
    plt.show()

def dict_to_df(dictionary):

    '''Return a dataframe from 'dictionary' '''
    df = pd.DataFrame(dictionary)
    return df

def value_counts_df(original_dataframe, column = None):

    '''This function outputs a dataframe with counts of the unique values for each column from the input dataframe (function argument).
    The counts are given for each column of the input dataframe - in the output a column with
    unique values is paired with another column with the counts for the unique values'''
    df_dict = dict()
    if column == None:
        for i in original_dataframe.columns:
            series_value_counts = original_dataframe[i].value_counts()
            dict_value_key = series_value_counts.name+"_value"
            dict_count_key = series_value_counts.name+"_count"
            df_dict[dict_value_key] = list(series_value_counts.index)
            df_dict[dict_count_key] = list(series_value_counts.values)
    else:
        i = column
        series_value_counts = original_dataframe[i].value_counts()
        dict_value_key = series_value_counts.name+"_value"
        dict_count_key = series_value_counts.name+"_count"
        df_dict[dict_value_key] = list(series_value_counts.index)
        df_dict[dict_count_key] = list(series_value_counts.values)

    return pd.DataFrame(collections.OrderedDict([(key,pd.Series(values)) for key,values in df_dict.items()]))


def histogram( two_dim_list, x_axis_titles, legend_labels, x_label, y_label, graph_title, text_size = 11, axesfont = 26, titlesize = 32, opacity = .6  ):

    '''Prints a histogram: two_dim_list variable should contain n-number (number of differnt colored series to plot) of lists of numerical values l-length long.
    x_axis_titles is a list of strings l-length long
    legend labels is a list of strings to the number of lists contained in two_dim_list
    ex: two_dim_list = [ [.1, .4, .3, .2], [.2, .4, .3, .1], [.2, .3, .3, .2] ], x_axis_titles = ['Pepperoni','Sausage','Cheese','Vegetable'], legend_labels = ['under 20 years old','20-50','50+']
    '''


    division_list = two_dim_list
    all_col = x_axis_titles

    plt.style.use('ggplot')

    n_groups = len(all_col)
    x_index_list = all_col
    x_axis_ticklabels = all_col

    fig, ax = plt.subplots()
    index = np.arange(n_groups)

    bar_width = ( 1 / len(division_list) ) - ( .1 / len(division_list) )
    colors = ['r','g','b','c','y']


    for i in range(len(division_list)):

        a = ax.bar(index + bar_width * i, division_list[i], bar_width, alpha = opacity, color = colors[i], label = legend_labels[i])


    plt.xlabel(x_label, weight = 'bold', fontsize = axesfont)
    plt.ylabel(y_label, weight = 'bold', fontsize = axesfont)
    plt.suptitle(graph_title, weight = 'bold', fontsize = titlesize )
    ax.set_xticks(index + bar_width / 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(text_size)

    #for i in range(len(x_axis_ticklabels)):
    #    x_axis_ticklabels[i] = '\n'.join(x_axis_ticklabels[i].split(':'))

    ax.set_xticklabels( x_axis_ticklabels )
    ax.legend()

    plt.show()
