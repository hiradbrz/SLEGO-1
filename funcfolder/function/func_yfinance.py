import json
from io import BytesIO
import boto3
import pandas as pd
from io import StringIO
import datetime
import yahoo_fin.stock_info as si
import ast

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


def import_new_yahoo_csv_s3(ticker:str='msft', 
                            output_s3_file_key:str ='data/yfinance_news.csv' ):                   
    """
    This function fetches historical market data for a given stock ticker from Yahoo Finance for a specified date range, and stores it as a CSV file in an Amazon S3 bucket.

    Parameters:
    ticker (str): The stock ticker symbol for which historical market data is to be fetched. Default is 'msft' for Microsoft.
    start_date (str): The start date for the period for which data should be fetched. The date should be specified in the format "YYYY-MM-DD". Default is one year prior to the current date.
    end_date (str): The end date for the period for which data should be fetched. The date should be specified in the format "YYYY-MM-DD". Default is the current date.
    output_s3_file_key (str): The destination where the fetched market data should be stored in the S3 bucket. The data is stored as a CSV file. Default location is 'data/yfinance_ohlc.csv'.

    Returns:
    df (DataFrame): A pandas DataFrame containing the fetched historical market data.

    The DataFrame and the CSV file contain the following columns:
    'open': The opening price for each trading day.
    'high': The highest price for each trading day.
    'low': The lowest price for each trading day.
    'close': The closing price for each trading day.
    'adjclose': The adjusted closing price for each trading day.
    'volume': The trading volume for each trading day.
    'ticker': The ticker symbol of the stock.
    'Datetime': The date of each trading day.
    """
    from yahoo_fin import news
    news_list  = news.get_yf_rss(ticker)
    df = pd.json_normalize(news_list)
    df['Datetime'] = pd.to_datetime(df['published']).astype(str)
    df.index = df['Datetime']
    # Upload the file
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket = bucket_name, Key = output_s3_file_key)
    return df

def import_marketdata_yahoo_csv_2_s3(ticker:str='msft', start_date:str =  (datetime.datetime.now()- datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                                            end_date:str = datetime.datetime.now().strftime("%Y-%m-%d"), 
                                            output_s3_file_key:str ='data/yfinance_ohlc.csv' ):                   

    df = si.get_data(ticker , start_date = start_date , end_date = end_date)
    df['Datetime'] = df.index
    # Upload the file
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket = bucket_name, Key = output_s3_file_key)
    return df

def calculate_daily_return_csv(input_s3_file_key:str='data/input.csv',
                          output_s3_file_key:str= 'data/output.csv',
                          return_period:int=2,
                          keep_original_columns:bool=True,
                          keep_return_period_rows:bool=False,
                          affected_columns:str=['close'],
                          ):
  
    df = __download_s3_file_as_df(input_s3_file_key=input_s3_file_key)

    df_ret = df[affected_columns].pct_change(return_period)
    df_ret = df_ret.add_suffix('_ret_'+str(return_period))

    if keep_original_columns:
        df = pd.concat([df, df_ret], axis=1)
    else:
        df = df_ret

    if not keep_return_period_rows:
        df = df.iloc[return_period:]

    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket = bucket_name, Key = output_s3_file_key)
    return df

