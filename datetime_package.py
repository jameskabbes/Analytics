import CustomPandas as cpd
import datetime

def get_datetime_input():

    '''inputs designed for specifying a date'''

    year = get_int_input('Enter year: ', 0, 3000)
    month = get_int_input('Enter month: ', 1, 12)
    day = get_int_input('Enter day: ', 1, 31)
    hour = get_int_input('Enter hour: ',  0, 23)
    minute = get_int_input('Enter minute: ', 0, 59)
    second = get_int_input('Enter second: ', 0, 59)

    return datetime.datetime(year, month, day, hour = hour, minute = minute, second = second)

def get_weekday(df, col, export_col = 'WEEKDAY'):
    '''Exports df with column export_col correspondign to the zero-based index of weekday
    '''
    # 0       1      2     3     4    5     6
    # Monday, Tues, Wed, Thurs, Fri, Sat, Sun
    df[export_col] = df[col].apply(lambda x: x.weekday())

    return df

def get_week_num(df, col, export_col = 'WEEK'):

    #1 based index
    #52 weeks in a year
    df[export_col] = df[col].apply( lambda x: x.isocalendar()[1] )
    return df
