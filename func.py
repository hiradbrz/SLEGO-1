import json
from io import BytesIO
import boto3
import pandas as pd
from io import StringIO
import datetime
import yahoo_fin.stock_info as si
import ast
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json
import boto3
from io import BytesIO
from pandas.api.types import is_number
'''
Result will be saved in current path as a csv file 
'''
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go

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

def __save_json_to_s3(jsonfile:dict=None, s3_file_key_output:str='data/output.json'):
    jsonfile = json.dumps(jsonfile)
    s3.put_object(Body=jsonfile, Bucket=bucket_name, Key=s3_file_key_output)
    return None

def convert_yfin_ohlc_2_adage3(input_s3_file_key:str='data/input.csv',
                          output_s3_file_key:str= 'data/output.json',
                          event_type:str='stock_ohlc',
                          time_column:str='Datetime',
                          timezone:str='GMT+11',
                          duration:int=1,
                          duration_unit:str='day',
                          ):
  
  df = __download_s3_file_as_df(input_s3_file_key=input_s3_file_key)

  ohlc = df
  ohlc[time_column] = pd.to_datetime(ohlc[time_column], format='%Y-%m-%d').astype(str)
  ohlc_json = ohlc.to_json(orient="records")
  ohlc_json = ast.literal_eval(ohlc_json)


  now= datetime.datetime.now()
  date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")

  # Our data model
  jsonfile_ohlc = {"data_source":"yahoo_finance", 
                    "dataset_type": "Daily stock price", 
                    "dataset_id": "http://bucket-name.s3-website-Region.amazonaws.com", 
                  "time_object":{"timestamp":date_time, "timezone":timezone},
                    "events":[]}

  for i in range(len(ohlc_json)):
    datetime_object = pd.to_datetime(ohlc_json[i][time_column], format='%Y-%m-%d %H:%M:%S.%f' )
    datetime_object= datetime_object.strftime('%Y-%m-%d %H:%M:%S.%f')
    jsonfile_ohlc['events'].append({"time_object":{"timestamp":datetime_object, "duration":duration, "duration_unit":duration_unit,"timezone":timezone},
                                    "event_type":event_type,   
                                    "attribute":ohlc_json[i] })


  __save_json_to_s3(jsonfile_ohlc, s3_file_key_output=output_s3_file_key)
  return jsonfile_ohlc


def validate_adage3_json_schema_s3(input_s3_file_key:str='data/jsonfile_ohlc.json'):

    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read().decode('utf-8')
    # Load the content into a JSON
    data = json.loads(file_content)

    schema = {
    "type": "object",
    "properties": 
    {
        "data_source": {"type": "string"},
        "dataset_type": {"type": "string"},
        "dataset_id": {"type": "string"},
        
        "time_object": {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string"},
            "timezone": {"type": "string"}},
        "required": ["timestamp","timezone"]
                        },
        "events": {"type": "array",
                "items": {"$ref":"#/$defs/events"},
                "minItems": 1,
        }
    },

        "required": ["data_source","dataset_id", "dataset_type","time_object","events"],

        "$defs": {
            "events": {
                "type": "object",
                "properties": {
                    "time_object": {"type": "object",
                                    "properties": {
                                        "timestamp":{"type": "string"},
                                        "duration":{"type": "number"},
                                        "timezone":{"type": "string"},
                                        "duration_unit":{"type": "string"},
                                                    },
                    
                                    "required": ["timestamp","duration","duration_unit","timezone"],       
                                    },
                        
                    "event_type": {"type": "string"},
                    "attribute": {"type": "object"},
                    },
                "required": ["time_object","event_type","attribute"],
                },
        }
    }

    try:
        validate(instance=data, schema=schema)
        return 'Valid data'
    except ValidationError as e:
        return 'Invalid data: ' + str(e)
    


def visualize_csv_lineplot(input_s3_file_key:str='data/inout.csv', 
                             output_s3_file_key:str='data/output.html'):

    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read()
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0)

    timestamps = df.index
    df['timestamps'] = timestamps
    df.insert(0, 'timestamps', df.pop('timestamps'))
    col_names = list(df.columns)
    coltypes = list(df.dtypes)

    specs = [[{"type": "table"}]]

    for col in col_names:
        specs.append([{"type": "scatter"}])

    fig = make_subplots(
        rows=df.shape[1] + 1, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        subplot_titles=['timestamps'] + col_names,
        specs=specs)

    fig.add_trace(go.Table(header=dict(values=col_names, font=dict(size=10), align="left"),
                           cells=dict(values=[df[k].tolist() for k in df.columns], align="left")), row=1, col=1)

    for i in range(len(col_names)):
        if coltypes[i] == 'O':
            y_out = pd.to_numeric(df[col_names[i]], errors='ignore', downcast=None)
            if not is_number(y_out):
                y_out = df[col_names[i]].apply(len)
            fig.add_trace(go.Scatter(x=timestamps, y=y_out, hovertemplate=df[col_names[i]], name=col_names[i]),
                          row=2 + i, col=1)
        else:
            fig.add_trace(go.Scatter(x=timestamps, y=df[col_names[i]], name=col_names[i]), row=2 + i, col=1)

    # Save the plot to HTML string
    fig.update_layout(height=250 * len(col_names), title_text="Visualized ADAGE3 data", legend=dict(orientation="h"))
    html_string = fig.to_html(full_html=False)

    # Upload the HTML string to S3
    s3.put_object(Body=html_string, Bucket=bucket_name, Key=output_s3_file_key)

    return df

def convert_adage3_attribute_json2csv(input_s3_file_key:str='data/input.json', 
                                     output_s3_file_key:str='data/output.csv'):

    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read().decode('utf-8')
    # Load the content into a JSON
    data = json.loads(file_content)

    list_att=[]
    list_time=[]

    for i in range(len(data['events'])):
        list_att.append(data['events'][i]['attribute'])
        list_time.append(data['events'][i]['time_object']['timestamp'])

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(list_att)

    timestamp_strings = list_time
    timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
    #timestamps = [datetime.strptime(ts, timestamp_format) for ts in timestamp_strings]
    try:
        timestamps = [datetime.strptime(ts, timestamp_format) for ts in timestamp_strings]
    except:
        timestamps = timestamp_strings    
    df.index = timestamps
  
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_s3_file_key)

    return df

def visualise_adage3_json(input_s3_file_key:str='data/input.json',output_s3_file_key:str='data/output.html'):
    '''
    Result will be saved in data folder as a html file 
    '''
    convert_adage3_attribute_json2csv(input_s3_file_key=input_s3_file_key, 
                                     output_s3_file_key='data/temp.csv')
    
    visualize_csv_lineplot(input_s3_file_key='data/temp.csv', 
                         output_s3_file_key=output_s3_file_key)

    return "html file saved in the data folder"


def convert_yfin_news_2_adage3(input_s3_file_key:str='data/yfinance_news.csv',
                          output_s3_file_key:str= 'data/output.json',
                          event_type:str='stock_news',
                          time_column:str='Datetime',
                          timezone:str='GMT+11',
                          duration:int=0,
                          duration_unit:str='NA',
                          ):
  
  df = __download_s3_file_as_df(input_s3_file_key=input_s3_file_key)

  df[time_column] = pd.to_datetime(df.index,format='%Y-%m-%d %H:%M:%S%z').astype(str)
  list_of_dict = df.to_dict('records')

  now= datetime.datetime.now()
  date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")

  # Our data model
  jsonfile_ohlc = {"data_source":"yahoo_finance", 
                    "dataset_type": "Daily stock news", 
                    "dataset_id": "http://bucket-name.s3-website-Region.amazonaws.com", 
                  "time_object":{"timestamp":date_time, "timezone":timezone},
                    "events":[]}

  for i in range(len(df)):
    datetime_object = df[time_column][i]
    #datetime_object= datetime_object.strftime('%Y-%m-%d %H:%M:%S.%f')
    jsonfile_ohlc['events'].append({"time_object":{"timestamp":datetime_object, "duration":duration, "duration_unit":duration_unit,"timezone":timezone},
                                    "event_type":event_type,   
                                    "attribute":list_of_dict[i] })


  __save_json_to_s3(jsonfile_ohlc, s3_file_key_output=output_s3_file_key)
  return jsonfile_ohlc


# a function install python package
def install(package:str = 'pandas'):
    # use pip to install package
    # pip install package
    import pip
    pip.main(['install', package])

import boto3
from io import BytesIO
import pandas as pd
from io import StringIO
import webbrowser
import dtale

def __login_aws():
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')
    bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
    folder_name = 'func' 
    return s3, bucket_name, folder_name

s3, bucket_name, folder_name = __login_aws()

def display_df_with_dtale(input_s3_file_key:str = 'data/yfinace_ohlc.csv'):
    # Create an S3 client
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read()
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0)
    # Start a D-Tale instance with your data
    d = dtale.show(df,app_root="/proxy")
    message = 'Change the poxy number at the end of the url to see the dtale dashboard:'+ str(d._url)
    return message


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



def test225(input:str = 'Hello'):
    return input

from io import BytesIO
import boto3
import pandas as pd
from io import StringIO
import datetime
import yahoo_fin.stock_info as si


def __login_aws():

    # Create a session using your AWS credentials
    s3 = boto3.client('s3')
    bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
    folder_name = 'func' 
    return s3, bucket_name, folder_name

s3, bucket_name, folder_name = __login_aws()


       
from io import BytesIO
import plotly.io as pio
import vectorbt as vbt
import pandas as pd
import datetime
import boto3
from io import BytesIO
import boto3
import pandas as pd
from io import StringIO

def backtest_MACorss_strategy( input_s3_file_key:str = 'data/yfinace_ohlc.csv',
                data_col:str = 'close',
                ma_fast:float = 10,
                ma_slow:float = 50,
                output_stats:bool = True,
                output_stats_s3_file_key:str ='data/output_stats.csv',
                output_plot:bool = True,
                output_plot_s3_file_key:str ='data/output_plot.html',
                output_position_record:bool = True,
                output_position_record_s3_file_key:str ='data/output_position_record.csv',
                output_return_record:bool = True,
                output_return_record_s3_file_key:str ='data/output_return_record.csv',
                return_period:str = 'd',
                ):
        """
        The function backtest() appears to be a backtesting strategy for a trading system based on moving averages, specifically using the crossover of a fast moving average (MA) and a slow moving average.

        The parameters for the function are as follows:

        ticker - This is the symbol for the stock that you are backtesting. The default is AAPL for Apple Inc.

        start - This is the start date for the backtesting period in 'YYYY-MM-DD' format. The default is '2019-03-01'.

        end - This is the end date for the backtesting period in 'YYYY-MM-DD' format. The default is '2023-09-01'.

        data_col - This is the column of data that the strategy should use from the downloaded data. The default is 'Close' which represents the closing price.

        ma_fast - This is the period for the fast moving average. The default is 10 days.

        ma_slow - This is the period for the slow moving average. The default is 50 days.

        output_stats - If set to True, the function will output the backtest statistics to a csv file. The default is True.

        output_plot - If set to True, the function will output a plot of the backtest. The default is True.

        output_position_record - If set to True, the function will output a record of the positions held during the backtest to a csv file. The default is True.

        In the function, vbt.YFData.download() is used to download historical stock price data from Yahoo Finance. This data is used to calculate the moving averages with vbt.MA.run().

        Entries and exits for the strategy are determined by the crossover of the fast and slow moving averages. When the fast MA crosses above the slow MA, this is considered an entry signal. When the fast MA crosses below the slow MA, this is considered an exit signal.

        The vbt.Portfolio.from_signals() function is then used to simulate the trading based on these entry and exit signals, assuming a daily frequency ('d').

        The backtest results are stored in the res object and the statistics of the backtest results are stored in the stats object. If the output_stats, output_plot, and output_position_record parameters are set to True, these results are saved to csv and html files respectively.

        At the end, the function returns the backtest statistics.

        Lastly, the function is called for the stock TSLA (Tesla Inc.) with a fast MA period of 10 days and a slow MA period of 50 days. The 'Close' price is used for the backtest calculations.
        """
        # Download the file
        response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
        # Get the file content
        file_content = response['Body'].read()
        # Load the content into a DataFrame
        df = pd.read_csv(BytesIO(file_content), index_col=0)

        data_close = df[data_col]
        ma_fast = vbt.MA.run(data_close, ma_fast)
        ma_slow = vbt.MA.run(data_close, ma_slow)
        
        entries = ma_fast.ma_crossed_above(ma_slow)
        exits   = ma_fast.ma_crossed_below(ma_slow)
        ma_fast

        res = vbt.Portfolio.from_signals(
                    data_close, 
                    entries = entries, 
                        exits = exits, 
                        freq = 'd')

        stats = res.stats()
        if output_stats:
                csv_buffer = StringIO()
                stats.to_csv(csv_buffer)
                s3.put_object(Body=csv_buffer.getvalue(), Bucket = bucket_name, Key = output_stats_s3_file_key)
        if output_plot:
                fig = res.plot(subplots = ['cum_returns', 'orders', 'trade_pnl'])
                out_file = BytesIO()
                html_str = pio.to_html(fig)
                out_file.write(html_str.encode())
                out_file.seek(0)  # Important: reset the position to the beginning of the file.
                s3.put_object(Body=out_file.read(), Bucket=bucket_name, Key=output_plot_s3_file_key, ContentType='text/html')
        if output_position_record:
                posrec= res.positions.records_readable
                csv_buffer2 = StringIO()
                posrec.to_csv(csv_buffer2)
                s3.put_object(Body=csv_buffer2.getvalue(), Bucket = bucket_name, Key = output_position_record_s3_file_key)
        if output_return_record:
                rets = res.returns()
                csv_buffer3 = StringIO()
                rets.to_csv(csv_buffer3)
                s3.put_object(Body=csv_buffer3.getvalue(), Bucket = bucket_name, Key = output_return_record_s3_file_key)

        return stats




