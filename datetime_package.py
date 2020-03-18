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

def prep_datetime(df, time_col, dt_col, format = '%Y-%m:-%d %H:%M:%S'):

    df[dt_col] = df[time_col].apply(lambda x: datetime.strptime(x, format) )
    return df

def datetime_to_string(dt, format = 'YYYY-MM-DD HH-MM-SS', perc_format = '%Y-%m-%d %H:%M:%S'):

    '''converts datetime to string'''
    if format == 'YYYY-MM-DD HH-MM-SS':
        a = dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        a = dt.strftime(perc_format)

    return a

def adjust_mismatched_time_series(df_to_keep, df_to_adjust, df1_dt_col, df2_dt_col, adjust_for_all_entries = True, how = '>='):

    df_to_keep = cpd.sort_df(df_to_keep, df1_dt_col)
    df_to_adjust = cpd.sort_df(df_to_adjust, df2_dt_col)

    og_datetimes = list(df_to_keep[df1_dt_col])

    print (df_to_keep)
    print (df_to_adjust)

    adjusted_cols = list(df_to_adjust.columns)
    dt_ind = adjusted_cols.index( df2_dt_col )
    dt2_new_dt_col = 'Sampled' + params.split + df2_dt_col
    adjusted_cols.insert(0, dt2_new_dt_col)

    rows = np.array([ [0,] * len(adjusted_cols) ] )
    other_ind = 0

    for og_dt_num in range(len(og_datetimes)):

        og_dt = og_datetimes[og_dt_num]
        try:
            og_dt2 = og_datetimes[og_dt_num + 1]
        except:
            og_dt2 = None

        for i in range(other_ind, len(df_to_adjust)):
            add = False
            br = False
            dt = df_to_adjust.loc[i, df2_dt_col]

            try:
                dt2 = df_to_adjust.loc[i+1, df2_dt_col]
            except:
                dt2 = None

            ###Add dt to the list if og_dt is >= to the dt value
            if how == '>=':
                if og_dt < dt:
                    if i == 0:
                        if adjust_for_all_entries:
                            add = True
                            br = True

                else: #og_dt >= dt
                    br = True
                    add = True

                    if dt2 != None:
                        if og_dt >= dt2:
                            other_ind += 1
                            continue

            elif how == '>':
                if og_dt <= dt:
                    if i == 0:
                        if adjust_for_all_entries:
                            add = True
                            br = True

                else: #og_dt >= dt
                    br = True
                    add = True

                    if dt2 != None:
                        if og_dt > dt2:
                            other_ind += 1
                            continue

            elif how == '<=':

                if og_dt > dt:
                    if i == len(df_to_adjust) - 1:
                        if adjust_for_all_entries:
                            add = True
                            br = True
                    else:
                        other_ind += 1

                else: #og_dt >= dt
                    br = True
                    add = True


            elif how == '<':

                if og_dt >= dt:
                    if i == len(df_to_adjust) - 1:
                        if adjust_for_all_entries:
                            add = True
                            br = True
                    else:
                        other_ind += 1

                else: #og_dt > dt
                    br = True
                    add = True


            if add:
                row = list(df_to_adjust.loc[i])
                row.insert(0, og_dt)
                rows = np.append(rows, [ row ], axis = 0)

            if br:
                break

    #remove the first row
    rows =  rows[1: ,:]
    print (rows)
    print (adjusted_cols)

    new_dict = {}
    for i in range(len(adjusted_cols)):
        new_dict[ adjusted_cols[i] ] = list(rows[:,i])

    adjusted_df = pd.DataFrame(new_dict)
    return adjusted_df
