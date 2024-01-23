import json
from io import BytesIO
import boto3
import pandas as pd
from io import StringIO
import datetime
import yahoo_fin.stock_info as si
import ast
import numpy as np
from typing import List, Dict
import plotly.express as px
import statsmodels.api as sm

def __login_aws():
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')
    bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
    folder_name = 'func' 
    return s3, bucket_name, folder_name

s3, bucket_name, folder_name = __login_aws()

def __download_s3_file_as_df(input_s3_file_key:str='data/input.csv'):
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read()
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0)
    return df

def get_stock_data(tickers: list = ['MSFT'],
                   start_date: str = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                   end_date: str = datetime.datetime.now().strftime("%Y-%m-%d")) -> str:
    """
    Fetches stock data based on provided tickers and a date range, then uploads it to S3 both in CSV format.
    
    Parameters:
    - tickers (list): List of stock tickers. Default is ['MSFT'].
    - start_date (str): Start date for the data range in YYYY-MM-DD format. Default is 365 days from the current date.
    - end_date (str): End date for the data range in YYYY-MM-DD format. Default is the current date.

    Returns:
    - str: A message indicating the upload status and tickers processed.
    """
    s3 = boto3.client('s3')  
    
    for ticker in tickers:
        # Get stock data in DataFrame format
        df = si.get_data(ticker, start_date=start_date, end_date=end_date)
        
        # Save CSV data to S3
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        s3_csv_key = f"data/{ticker}_yfinance_ohlc.csv"
        s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=s3_csv_key)

    return f"Uploaded data for tickers: {', '.join(tickers)}"


def get_index_data(tickers: list = ['^GSPC'],
                   start_date: str = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                   end_date: str = datetime.datetime.now().strftime("%Y-%m-%d")) -> str:
    """
    Fetches index data based on provided index tickers and a date range, then uploads it to S3 CSV format.
    
    Parameters:
    - tickers (list): List of index tickers. Default is ['^GSPC'].
    - start_date (str): Start date for the data range in YYYY-MM-DD format. Default is 365 days from the current date.
    - end_date (str): End date for the data range in YYYY-MM-DD format. Default is the current date.

    Returns:
    - str: A message indicating the upload status and tickers processed.
    """
    
    s3 = boto3.client('s3')  
    
    for ticker in tickers:
        # Get index data in DataFrame format
        df = si.get_data(ticker, start_date=start_date, end_date=end_date)
        
        # Save CSV data to S3
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        s3_csv_key = f"data/{ticker}_Index_yfinance_ohlc.csv"
        s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=s3_csv_key)

    return f"Uploaded data for indices: {', '.join(tickers)}"

def compute_daily_measures(tickers: list = ['MSFT']):

    for ticker in tickers:
        input_s3_file_key = f"data/{ticker}_yfinance_ohlc.csv"
        
        # Download the DataFrame from S3
        df = __download_s3_file_as_df(input_s3_file_key)
        
        # Standardize column names to lower case
        df.columns = map(str.lower, df.columns)

        df['TrdVolume'] = df['volume']
        df['OpnPrice'] = df['open']
        df['ClsPrice'] = df['close']
        df['HghPrice'] = df['high']
        df['LowPrice'] = df['low']
        df['LReturn_ClsPrice'] = np.log(df.close) - np.log(df.close.shift(1))
        df['AReturn_ClsPrice'] = df.close.pct_change()
        df['DilFact'] = abs(df['close'] - df['adjclose'])
        df['AdjLReturn_ClsPrice'] = np.log(df['adjclose']) - np.log(df['adjclose'].shift(1))
        df['AdjAReturn_ClsPrice'] = df['adjclose'].pct_change()
        df['AveReturn_20_ClsPrice'] = np.log(df['close'][0:20]) - np.log(df['close'][0:20].shift(1))/20
        df['StddevReturn_20_ClsPrice'] = np.std(np.log(df['close'][0:20]) - np.log(df['close'][0:20].shift(1)))

        # Save as CSV to S3
        output_csv_file_key = f"data/{ticker}_Daily_Measures.csv"
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=True)
        s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_csv_file_key)
    
    return df,"Computed and saved measures for tickers: {}".format(", ".join(tickers))

def compute_index_measures(tickers: list = ['^GSPC']):

    for ticker in tickers:
        input_s3_file_key = f"data/{ticker}_Index_yfinance_ohlc.csv"
        
        # Download the DataFrame from S3
        df = __download_s3_file_as_df(input_s3_file_key)
        
        # Standardize column names to lower case
        df.columns = map(str.lower, df.columns)

        df['TrdVolume'] = df['volume']
        df['OpnPrice'] = df['open']
        df['ClsPrice'] = df['close']
        df['HghPrice'] = df['high']
        df['LowPrice'] = df['low']
        df['LReturn_ClsPrice'] = np.log(df.close) - np.log(df.close.shift(1))
        df['AReturn_ClsPrice'] = df.close.pct_change()
        df['DilFact'] = abs(df['close'] - df['adjclose'])
        df['AdjLReturn_ClsPrice'] = np.log(df['adjclose']) - np.log(df['adjclose'].shift(1))
        df['AdjAReturn_ClsPrice'] = df['adjclose'].pct_change()
        df['AveReturn_20_ClsPrice'] = np.log(df['close'][0:20]) - np.log(df['close'][0:20].shift(1))/20
        df['StddevReturn_20_ClsPrice'] = np.std(np.log(df['close'][0:20]) - np.log(df['close'][0:20].shift(1)))

        # Save as CSV to S3
        output_csv_file_key = f"data/{ticker}_Index_Daily_Measures.csv"
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=True)
        s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_csv_file_key)
    
    return "Computed and saved measures for index: {}".format(", ".join(tickers))

def create_event_set(tickers: list = ['MSFT'], 
                     index_ticker: str='^GSPC', 
                     event_date: str='2023-02-03', 
                     window_size: int=10, 
                     estimation_window_size: int=10,
                     output_s3_file_key: str='data/Event_Study_Result.csv'):
    """
    Perform an event study on a given list of tickers and compare with market index data.
    
    This function computes the event study by analyzing the daily returns of given tickers 
    around a specified event date and juxtaposes the results with the market index data. 
    The function returns a DataFrame which contains the event window data, estimation window data, 
    and relative days with respect to the event.
    
    Parameters:
        tickers (list): List of tickers to compute the event study on.
        index_ticker (str): Ticker symbol for the market index data.
        event_date (str): The event date in YYYY-MM-DD format.
        window_size (int): The size of the event window (days around the event date).
        estimation_window_size (int): The size of the estimation window to provide context.
        output_s3_file_key (str): S3 file key to save the combined event study result CSV.
    
    Returns:
        pd.DataFrame: A DataFrame containing the event window data, estimation window data, 
                      relative days, and the returns of tickers and the market index.
    """
    
    try:
        # Get index data
        index_data = __download_s3_file_as_df(f"data/{index_ticker}_Index_Daily_Measures.csv")
        index_data.index = pd.to_datetime(index_data.index)
        
        event_date_dt = pd.to_datetime(event_date)
        results = {"Date": index_data.index, "Index": index_data['AReturn_ClsPrice']}
    
        aligned_ticker_data = {}
        for ticker in tickers:
            df = __download_s3_file_as_df(f"data/{ticker}_Daily_Measures.csv")
            df.index = pd.to_datetime(df.index)
            aligned_df, _ = df.align(index_data, axis=0, join='inner')
            aligned_ticker_data[ticker] = aligned_df['AReturn_ClsPrice']
    
        combined_df = pd.DataFrame(aligned_ticker_data)
        combined_df['Index'] = index_data['AReturn_ClsPrice']
        
        total_window_size = window_size + estimation_window_size
        if event_date_dt in combined_df.index:
            start_idx = max(0, combined_df.index.get_loc(event_date_dt) - total_window_size + 1)
            end_idx = min(len(combined_df), combined_df.index.get_loc(event_date_dt) + window_size)
            
            result = combined_df.iloc[start_idx:end_idx].copy()
            result = result.iloc[1:]
            
            event_day_position = result.index.get_loc(event_date_dt)
            result['EventDateRef'] = range(-event_day_position, len(result) - event_day_position)
            
            csv_buffer = StringIO()
            result.to_csv(csv_buffer, index=True)
            s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_s3_file_key)

            print(f"Event study result saved as {output_s3_file_key} in S3.")
            return result
        else:
            print(f"Event date {event_date} is not found in the combined data.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def compute_expected_returns(data_file_key: str ='data/Event_Study_Result.csv' , 
                             output_s3_file_key: str = 'data/Expected_Returns.csv'):
    """
    Compute expected returns for a time series based on multiple ticker returns and index returns.
    
    Parameters:
        data_file_key (str): S3 file key for the data CSV.
        output_s3_file_key (str): S3 file key for the output CSV.
    
    Returns:
        pd.DataFrame: DataFrame with an additional column for expected returns.
    """
    
    return_index_col_name = 'Index'
    
    # Download the data from S3
    ts = __download_s3_file_as_df(data_file_key)
    
    # Ensure the DataFrame's date column is in datetime format
    ts.index = pd.to_datetime(ts.index)
    
    index_returns = ts[return_index_col_name]
    
    # Get the columns for the tickers
    ticker_cols = [col for col in ts.columns if col not in [return_index_col_name, 'EventDateRef']]

    # Convert returns to percentage
    # ts[ticker_cols] = ts[ticker_cols] * 100
    # ts[return_index_col_name] = ts[return_index_col_name] * 100

    # Defining the variables for the regression
    x = ts[ticker_cols] * 100
    y = index_returns * 100
    x = sm.add_constant(x) # Add constant term to the predictor variables

    # Performing the regression and fitting the model
    result = sm.OLS(y, x).fit()
    
    # Printing the summary table
    print(result.summary())
    
    # Extracting the coefficients and adjusting them
    df1 = pd.read_html(result.summary().tables[1].as_html(), header=0, index_col=0)[0]
    df1["coef"] = df1["coef"] / 100

    # Calculate the expected return using the regression coefficients
    ts["Return.Expected"] = df1["coef"].loc["const"] + ts[ticker_cols].mean(axis=1) * df1["coef"][1:].mean()
    
    # Save the resulting DataFrame to S3
    csv_buffer = StringIO()
    ts.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_s3_file_key)
    
    print(f"Result saved as {output_s3_file_key} in S3.")
    
    return ts

def compute_abnormal_returns(data_file_key: str ='data/Expected_Returns.csv', 
                             output_s3_file_key: str ='data/Abnormal_Returns.csv'):
    """
    Compute the abnormal returns for a time series based on the actual returns of tickers and the expected returns.
    
    Parameters:
        data_file_key (str): S3 file key for the data CSV with actual and expected returns.
        output_s3_file_key (str): S3 file key for the output CSV with abnormal returns.
    
    Returns:
        pd.DataFrame: DataFrame with an additional column for abnormal returns for each ticker.
    """

    # Download the data from S3
    df = __download_s3_file_as_df(data_file_key)
    
    # Extract the columns corresponding to the tickers
    tickers = [col for col in df.columns if col not in ['Index', 'EventDateRef', 'Date', 'Return.Expected']]
    
    for ticker in tickers:
        df[f'AbnormalReturn_{ticker}'] = df[ticker] - df['Return.Expected']

    # Save the resulting DataFrame to S3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_s3_file_key)

    print(f"Abnormal returns saved as {output_s3_file_key} in S3.")
    
    return df

def compute_mean_abnormal_returns(data_file_key: str = 'data/Abnormal_Returns.csv',
                                  output_s3_file_key: str = 'data/Mean_Abnormal_Returns.csv'):
    """
    Compute the mean abnormal returns based on a dataset containing abnormal returns for multiple tickers.
    
    Parameters:
        data_file_key (str): S3 file key for the data containing abnormal returns.
        output_s3_file_key (str): S3 file key for saving the computed mean abnormal returns.
    
    Returns:
        pd.DataFrame: DataFrame containing the mean abnormal returns for each ticker.
    """

    # Download the DataFrame containing abnormal returns
    df = __download_s3_file_as_df(data_file_key)
    
    # Identify columns corresponding to abnormal returns of tickers
    abnormal_return_cols = [col for col in df.columns if col.startswith('AbnormalReturn_')]
    
    if not abnormal_return_cols:
        raise ValueError("The provided dataset does not contain any 'AbnormalReturn_' columns.")
    
    # Compute the mean abnormal return for each ticker and store in a dictionary
    mean_abnormal_returns = {}
    for col in abnormal_return_cols:
        ticker = col.replace("AbnormalReturn_", "")
        mean_abnormal_returns[ticker] = df[col].mean()
    
    # Convert the dictionary to a DataFrame
    mean_df = pd.DataFrame(list(mean_abnormal_returns.items()), columns=["Ticker", "MeanAbnormalReturn"])
    
    # Save the mean abnormal return DataFrame to S3
    csv_buffer = StringIO()
    mean_df.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_s3_file_key)

    print(f"Mean abnormal returns saved as {output_s3_file_key} in S3.")
    
    return mean_df

def plot_abnormal_returns(
        input_s3_file_key: str = 'data/Abnormal_Returns.csv',
        output_s3_file_key: str = 'data/AbnormalReturns_plot.html'
):
    """
    Plot the abnormal returns for tickers over time and save the plot as an HTML file.
    
    Parameters:
        input_s3_file_key (str): S3 file key for the input data containing abnormal returns.
        output_s3_file_key (str): S3 file key for saving the plot as an HTML file.
    
    Returns:
        None
    """
    df = __download_s3_file_as_df(input_s3_file_key)
    
    # Concatenate EventDateRef and the DataFrame's index for the x-axis
    df['X-Axis'] = df.index.astype(str) + ' (' + df['EventDateRef'].astype(str) + ')'

    # Extract the tickers for which abnormal returns were computed
    abnormal_return_cols = [col for col in df.columns if col.startswith('AbnormalReturn_')]
    
    for col in abnormal_return_cols:
        fig = px.line(df, x='X-Axis', y=col, 
                      title=f'Abnormal Returns for {col.split("_")[1]} over Time',
                      labels={'X-Axis': 'Date (Event Date Ref)', col: 'Abnormal Return'})
        
        # Find the X-Axis value where EventDateRef equals to 0
        x_val_for_event_0 = df[df['EventDateRef'] == 0]['X-Axis'].iloc[0]
        
        # Add a vertical line
        fig.add_vline(x=x_val_for_event_0, line_color="red", line_dash="dash", line_width=2, name="EventDateRef = 0")

        # Adjust the output file name to differentiate between different tickers
        modified_output_s3_file_key = output_s3_file_key.replace('.html', f'_{col.split("_")[1]}.html')
        
        # Save the figure as an HTML file in memory
        html_buffer = StringIO()
        fig.write_html(html_buffer)
        html_buffer.seek(0)
        
        # Upload the HTML buffer to S3
        s3.put_object(Body=html_buffer.read(), Bucket=bucket_name, Key=modified_output_s3_file_key, ContentType='text/html')
        
        print(f"Abnormal returns plot for {col.split('_')[1]} saved as {modified_output_s3_file_key} in S3.")

def plot_accumulated_abnormal_returns(
        input_s3_file_key: str = 'data/Abnormal_Returns.csv',
        output_s3_file_key: str = 'data/AccumulatedAbnormalReturns_plot.html'
):
    """
    Plot the accumulated abnormal returns for tickers over time and save the plot as an HTML file.
    
    Parameters:
        input_s3_file_key (str): S3 file key for the input data containing abnormal returns.
        output_s3_file_key (str): S3 file key for saving the plot as an HTML file.
    
    Returns:
        None
    """
    df = __download_s3_file_as_df(input_s3_file_key)
    
    # Concatenate EventDateRef and the DataFrame's index for the x-axis
    df['X-Axis'] = df.index.astype(str) + ' (' + df['EventDateRef'].astype(str) + ')'

    # Extract the tickers for which abnormal returns were computed
    abnormal_return_cols = [col for col in df.columns if col.startswith('AbnormalReturn_')]

    # Compute the accumulated abnormal returns for each ticker
    for col in abnormal_return_cols:
        df[f"Accumulated_{col}"] = df[col].cumsum()

    accumulated_cols = [col for col in df.columns if col.startswith('Accumulated_AbnormalReturn_')]
    
    for col in accumulated_cols:
        fig = px.line(df, x='X-Axis', y=col, 
                      title=f'Accumulated Abnormal Returns for {col.split("_")[2]} over Time',
                      labels={'X-Axis': 'Date (Event Date Ref)', col: 'Accumulated Abnormal Return'})
        
        # Find the X-Axis value where EventDateRef equals to 0
        x_val_for_event_0 = df[df['EventDateRef'] == 0]['X-Axis'].iloc[0]
        
        # Add a vertical line
        fig.add_vline(x=x_val_for_event_0, line_color="red", line_dash="dash", line_width=2, name="EventDateRef = 0")

        # Adjust the output file name to differentiate between different tickers
        modified_output_s3_file_key = output_s3_file_key.replace('.html', f'_{col.split("_")[2]}.html')
        
        # Save the figure as an HTML file in memory
        html_buffer = StringIO()
        fig.write_html(html_buffer)
        html_buffer.seek(0)
        
        # Upload the HTML buffer to S3
        s3.put_object(Body=html_buffer.read(), Bucket=bucket_name, Key=modified_output_s3_file_key, ContentType='text/html')
        
        print(f"Accumulated abnormal returns plot for {col.split('_')[2]} saved as {modified_output_s3_file_key} in S3.")

def plot_mean_abnormal_returns(
        input_s3_file_key: str = 'data/Mean_Abnormal_Returns.csv',
        output_s3_file_key: str = 'data/Mean_AR_plot.html',
        ticker_column: str = 'Ticker',
        mean_abnormal_return_column: str = 'MeanAbnormalReturn'
):
    """
    Plot the mean abnormal returns for each ticker and save the plot as an HTML file.
    
    Parameters:
        input_s3_file_key (str): S3 file key for the input data containing mean abnormal returns.
        output_s3_file_key (str): S3 file key for saving the plot as an HTML file.
        ticker_column (str): Name of the column containing ticker symbols.
        mean_abnormal_return_column (str): Name of the column containing mean abnormal returns.
    
    Returns:
        None
    """

    df = __download_s3_file_as_df(input_s3_file_key)
    
    # Create the Plotly figure
    fig = px.bar(df, x=df.index, y=mean_abnormal_return_column,
                 title='Mean Abnormal Returns for Each Ticker',
                 labels={ticker_column: 'Ticker', mean_abnormal_return_column: 'Mean Abnormal Return'})

    # Save the figure as an HTML file in memory
    html_buffer = StringIO()
    fig.write_html(html_buffer)
    html_buffer.seek(0)
    
    # Upload the HTML buffer to S3
    s3.put_object(Body=html_buffer.read(), Bucket=bucket_name, Key=output_s3_file_key, ContentType='text/html')
    
    print(f"Mean abnormal returns plot saved as {output_s3_file_key} in S3.")
