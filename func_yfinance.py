
from io import BytesIO
import boto3
import pandas as pd
from io import StringIO
import datetime
import yahoo_fin.stock_info as si


def __login_aws(aws_access_key_id:str='AKIARK55CWM2F4ZFSKUB',
                aws_secret_access_key:str='JTObN613yPxw71VP68yl2TJ70amlHTNquzyXLnkE'):
    # Create a session using your AWS credentials
    s3 = boto3.client('s3', 
                        aws_access_key_id='AKIARK55CWM2F4ZFSKUB',
                        aws_secret_access_key='JTObN613yPxw71VP68yl2TJ70amlHTNquzyXLnkE')
    bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
    folder_name = 'func' 
    return s3, bucket_name, folder_name

s3, bucket_name, folder_name = __login_aws()

def __download_s3_file_as_df(input_s3_file_key:str='data/SCHW.csv'):
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read()
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0)
    return df
    
def __save_df_to_s3(df:pd.DataFrame=None, s3_file_key_output:str='data/output.csv'):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=s3_file_key_output)
    return None


def import_marketdata_yahoo_csv_2_s3(ticker:str='msft', start_date:str =  (datetime.datetime.now()- datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                                            end_date:str = datetime.datetime.now().strftime("%Y-%m-%d"), 
                                            output_s3_file_key:str ='data/yfinace_ohlc.csv' ):                   
    """ 
    This is the function that get yahoo finance market data!  
    output: csv table 
    columns: open,high,low,close,adjclose,volume,ticker,Datetime
    """

    df = si.get_data(ticker , start_date = start_date , end_date = end_date)
    df['Datetime'] = df.index
    # Upload the file
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket = bucket_name, Key = output_s3_file_key)
    return df

def yfinace_ohlc_return_s3data(input_s3_file_key:str='data/SCHW.csv', return_period:int= 1, return_col:str='adjclose' output_s3_file_key= 'data/SCHW_return.csv'):
    '''
    insert instruction 
    here
    '''
    
    df = __download_s3_file_as_df(input_s3_file_key='data/SCHW.csv')
    # Convert the DataFrame to numeric, turning non-numeric values into NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    df_return = df[return_col].pct_change(return_period)
    # df_return = df_return.dropna()
    __save_df_to_s3(df_return, s3_file_key_output=output_s3_file_key)
    return df_return


import pandas as pd
import quantstats as qs
from io import BytesIO
import datetime
import datetime
import yahoo_fin.stock_info as si

def import_marketdata_yahoo_csv_2_s3(ticker:str='msft', start_date:str =  (datetime.datetime.now()- datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                                            end_date:str = datetime.datetime.now().strftime("%Y-%m-%d"), 
                                            output_s3_file_key:str ='data/msft_ohlc.csv' ):                   
    """ 
    This is the function that get yahoo finance market data!  
    output: csv table 
    columns: open,high,low,close,adjclose,volume,ticker,Datetime
    """
    df = si.get_data(ticker , start_date = start_date , end_date = end_date)
    df['Date'] = df.index
    # Upload the file
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket = bucket_name, Key = output_s3_file_key)
    return df

def __load_s3_file_as_df(input_s3_file_key:str='data/msft_ohlc.csv'):
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read()
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0)
    return df

def __save_df_to_s3(df:pd.DataFrame=None, s3_file_key_output:str='data/output.csv'):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=s3_file_key_output)
    return None


def yfinace_ohlc_return_s3data(input_s3_file_key:str='data/SCHW.csv', return_period:int= 1, return_col:str='adjclose',
                                 output_s3_file_key:str= 'data/SCHW_return.csv'):
    '''
    insert instruction 
    here
    '''
    
    df = __download_s3_file_as_df(input_s3_file_key='data/SCHW.csv')
    # Convert the DataFrame to numeric, turning non-numeric values into NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    series_return = df[return_col].pct_change(return_period)
    
    df['return'] = series_return
    # df_return = df_return.dropna()
    __save_df_to_s3(df, s3_file_key_output=output_s3_file_key)
    return df_return

def generate_performance_report_from_return(s3_input_file_key:str='data/msft_ohlc.csv', 
                             col_name:str='return',
                            s3_output_file_key:str='data/quantstats-tearsheet.html',
                            report_title:str="Performance Report"):
    # Download the file
    df = __load_s3_file_as_df(input_s3_file_key=s3_input_file_key)
    # Convert the DataFrame to numeric, turning non-numeric values into NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    # Calculate the daily returns
    returns = df[col_name]

    # Generate the report
    qs.reports.html(returns, output='./datafolder/report.html', title=report_title)
    # Upload the file./
    s3.upload_file(Filename='./datafolder/report.html', Bucket=bucket_name, Key=s3_output_file_key)

    return None
