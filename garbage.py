import pandas as pd
import plotly.express as px
def is_zero(series_object, columns):
    # is_zero_list = []
    for i, value in enumerate(series_object):
        series_object[i] = value == 0

    return series_object

def display_zero_data(df):
    columns = df.columns
    percent_missing = 100 * df.copy().apply(lambda series: is_zero(series, columns), axis=1).sum()
    print(percent_missing)
    percent_missing = 100 * df.isnull().sum()
    print(percent_missing)
    # percent_missing /= len(df)
    # zero_value_df = pd.DataFrame({'column': df.columns,
    #                              'percent': percent_missing})
    # # print(zero_value_df.head())
    # zero_value_df.sort_values('percent', inplace=True)
    # zero_value_df.reset_index(drop=True, inplace=True)
    # zero_value_df = zero_value_df[zero_value_df['percent'] > 0]

    # Plotting
    # fig = px.bar(
    #     zero_value_df, 
    #     x='percent', 
    #     y="column", 
    #     orientation='h', 
    #     title='Columns with Values that Equal 0', 
    #     height=300, 
    #     width=600
    # )
    # fig.show()

if __name__ == "__main__":
    release_2010s = pd.read_csv("dbs/release_dates_2010s.csv").sort_values("month_released").head()
    display_zero_data(release_2010s)
