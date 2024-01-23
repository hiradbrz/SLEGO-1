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




import json
from io import BytesIO, StringIO
from typing import List, Dict
import boto3
import pandas as pd
import datetime
import yahoo_fin.stock_info as si
import ast
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import quantstats as qs
import plotly.io as pio
import vectorbt as vbt
from pandas.api.types import is_number
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt
from backtrader.feeds import PandasData
import pyfolio as pf
import matplotlib.pyplot as plt
from typing import List
import yfinance as yf

from time import gmtime, strftime
import math

def __login_aws():
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')
    bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
    folder_name = 'func' 
    return s3, bucket_name, folder_name

s3, bucket_name, folder_name = __login_aws()

def __download_s3_file_as_df(input_s3_file_key:str='data/input.csv'):
    s3, bucket_name, folder_name = __login_aws()
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read()
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0)
    return df

def __download_s3_FAIC_json_file(input_s3_file_key: str = 'data/input.json'):
    s3, bucket_name, folder_name = __login_aws()
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    # Get the file content
    file_content = response['Body'].read().decode('utf-8')
    
    # Remove the extra quotes and unescape the JSON string
    file_content = file_content.strip('"').replace('\\"', '"')
    
    # Load the content into a DataFrame
    try:
        df = pd.read_json(StringIO(file_content), orient='split')
    except ValueError as e:
        print(f"Error while reading JSON: {e}")
        return None
    
    return df

def __save_json_to_s3(jsonfile:dict=None, s3_file_key_output:str='data/output.json'):
    jsonfile = json.dumps(jsonfile)
    s3.put_object(Body=jsonfile, Bucket=bucket_name, Key=s3_file_key_output)
    return None

def __convert_df_to_json_model(df: pd.DataFrame, 
                               event_type: str = 'intraday_timeseries', 
                               timezone: str = 'GMT+11') -> Dict:
    """
    Convert a DataFrame to a specified JSON data model.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to convert
    - event_type (str): The event type in the JSON model
    - timezone (str): The timezone for the time data
    
    Returns:
    - Dict: The constructed JSON data model
    """
    # Create JSON data model
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    json_data_model = {
        "data_source": "Reuters",
        "dataset_type": "Intraday Timeseries",
        "dataset_id": "http://bucket-name.s3-website-Region.amazonaws.com",
        "time_object": {"timestamp": date_time, "timezone": timezone},
        "events": []
    }

    # Convert DataFrame to JSON
    df_json = df.to_json(orient="records")
    df_json = json.loads(df_json)

    # Populate the JSON data model
    for record in df_json:
        datetime_object = "1970-01-01 00:00:00.000000"  # Default value
        if 'Date-Time' in record:
            datetime_object = pd.to_datetime(record['Date-Time'], format='%Y-%m-%d %H:%M:%S')
            datetime_object = datetime_object.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        json_data_model['events'].append({
            "time_object": {"timestamp": datetime_object, "timezone": timezone},
            "event_type": event_type,
            "attribute": record
        })
    
    return json_data_model

def __convert_yfin_ohlc_2_adage3(input_s3_file_key:str='data/input.csv',
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

def build_intraday_timeseries(input_s3_file_key:str='data/BHPAX_20190717_Allday.csv',
                                        return_period:int=2,
                                        keep_original_columns:bool=True,
                                        keep_return_period_rows:bool=False,
                                        affected_columns:str=['Close'],
                                        output_csv_file_key='data/Intraday_Timeseries.csv',
                                        output_json_file_key='data/Intraday_Timeseries.json'):
    
    # Fetch the desired file from S3
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    file_content = response['Body'].read()
    df = pd.read_csv(BytesIO(file_content), index_col=0)

    # Data Preprocessing
    df = df[['Date-Time','Type','Price','Volume']]
    df.dropna(subset=['Price'], inplace=True)
    df['PV'] = df['Price'] * df['Volume']

    # Vectorized operation to modify 'Date-Time' column
    df['Date-Time'] = df['Date-Time'].str[0:10] + ' ' + df['Date-Time'].str[11:19]

    # Placeholder for return calculation (df_ret), since it is not defined in the original code
    df_ret = pd.DataFrame()

    # Decide whether to keep the original columns in the resulting DataFrame
    if keep_original_columns:
        df = pd.concat([df, df_ret], axis=1)
    else:
        df = df_ret

    # Decide whether to drop the initial rows up to the return period
    if not keep_return_period_rows:
        df = df.iloc[return_period:]

    # Save as CSV to S3
    # csv_buffer = StringIO()
    # df.to_csv(csv_buffer, index=False)
    # s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_csv_file_key)

    # Save as JSON to S3
    json_data = __convert_df_to_json_model(df)
    __save_json_to_s3(json_data, output_json_file_key)
    
    print(f"Data saved as {output_json_file_key} in S3")
    return df

def compute_intraday_measures(input_s3_file_key: str = 'data/Intraday_Timeseries.json',
                              aggregation_period: int = 1,
                              affected_columns: List[str] = ['Price', 'Volume'],
                              output_csv_file_key: str = 'data/Intraday_Measures.csv',
                              output_json_file_key: str = 'data/Intraday_Measures.json') -> pd.DataFrame:

    # Fetch data from S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    file_content = response['Body'].read().decode('utf-8')
    json_data = json.loads(file_content)
    
    # Convert JSON to DataFrame
    events = json_data.get('events', [])
    df = pd.DataFrame([event['attribute'] for event in events])
    
    # Preprocess the DataFrame
    required_columns = ['Date-Time', 'Type'] + affected_columns
    df = df[required_columns]
    df = df[df['Type'] == 'Trade']
    df.dropna(subset=['Price'], inplace=True)
    df['PV'] = df['Price'] * df['Volume']
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    
    # Calculate intraday measures
    df_grouped = df.groupby(pd.Grouper(key='Date-Time', freq=f"{aggregation_period}T"))
    new_df = pd.DataFrame({
        'TradeCount': df_grouped.size(),
        'AveragePrice': df_grouped['Price'].mean(),
        'DollarVolumeTraded': df_grouped['PV'].sum(),
        'ShareVolumeTraded': df_grouped['Volume'].sum(),
        'Close': df_grouped['Price'].last()  
    }).reset_index()
    
    new_df.dropna(inplace=True)
    new_df['VWAP'] = new_df['DollarVolumeTraded'] / new_df['ShareVolumeTraded']
    new_df['AReturn-VWAP'] = new_df['VWAP'].pct_change()
    new_df['LReturn-VWAP'] = np.log(new_df['VWAP']) - np.log(new_df['VWAP'].shift(1))
    

    # Save as CSV to S3
    csv_buffer = StringIO()
    new_df.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_csv_file_key)

    # Save as JSON to S3
    json_data = new_df.to_json(orient='split')
    __save_json_to_s3(json_data, output_json_file_key)
    
    print(f"Data saved as {output_json_file_key} in S3")
    return new_df

def generate_MA_trading_signals(input_s3_file_key="data/Intraday_Measures.json", 
                                    output_csv_file_key='data/ma_output.csv',
                                    output_json_file_key='data/ma_output.json',
                                    short_window=20, 
                                    long_window=50):
    # Download the input data
    data = __download_s3_FAIC_json_file(input_s3_file_key=input_s3_file_key)

    # Check if 'Close' column exists in the data
    if 'Close' not in data.columns:
        raise ValueError("The data must contain a 'Close' column.")
    
    close_price_df = data['Close']
    date = close_price_df.index

    # Compute short-term and long-term moving averages
    short_moving_avg = close_price_df.rolling(window=short_window, min_periods=1).mean()
    long_moving_avg = close_price_df.rolling(window=long_window, min_periods=1).mean()

    # Create DataFrame to store the data and moving averages
    signals_df = pd.DataFrame({
        'Close': close_price_df,
        'Short_MA': short_moving_avg,
        'Long_MA': long_moving_avg
    })

    # Create the trading signals based on the crossover strategy
    signals_df['Signal'] = 0
    signals_df.loc[short_moving_avg > long_moving_avg, 'Signal'] = 1
    signals_df.loc[short_moving_avg < long_moving_avg, 'Signal'] = -1
    
    # Save as CSV to S3
    csv_buffer = StringIO()
    signals_df.to_csv(csv_buffer, index=True)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_csv_file_key)

    # Save as JSON to S3
    json_data_model = __convert_df_to_json_model(signals_df, event_type="ma_trading_signals")
    __save_json_to_s3(json_data_model, output_json_file_key)
    
    print(f"Data saved as {output_json_file_key} in S3")

def generate_ML_trading_signals(input_s3_file_key="data/Intraday_Measures.json", 
                                output_csv_file_key='data/ml_output.csv',
                                output_json_file_key='data/ml_output.json'):
    # Download the input data
    data = __download_s3_FAIC_json_file(input_s3_file_key=input_s3_file_key)

    # Check if 'close' column exists in the data
    if 'Close' not in data.columns:
        raise ValueError("The data must contain a 'close' column.")

    # Feature Engineering
    data['Short_MA'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['Price_Delta'] = data['Close'].diff()

    # Drop NA values generated by rolling mean and diff
    data = data.dropna()

    # Prepare features and labels
    X = data[['Close', 'Short_MA', 'Long_MA', 'Price_Delta']]
    y = np.sign(data['Price_Delta'].shift(-1))  # Next movement (up=1, down=-1, stable=0)
    y = y.dropna()
    X = X.iloc[:-1]  # Make the features array one shorter to align with y

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Generate trading signals on the entire dataset
    signals = clf.predict(X)
    data = data.iloc[:-1]  # Trim the DataFrame to match the length of signals
    data['Signal'] = signals

    # Save as CSV to S3
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=True)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_csv_file_key)

    # Save as JSON to S3
    json_data = data.to_json(orient='split')
    __save_json_to_s3(json_data, output_json_file_key)
    
    print(f"Data saved as {output_json_file_key} in S3")

def generate_LSTM_trading_signals(input_s3_file_key="data/Intraday_Measures.json", 
                                  output_csv_file_key='data/lstm_output.csv',
                                  output_json_file_key='data/lstm_output.json'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # Download and check data
    data = __download_s3_FAIC_json_file(input_s3_file_key=input_s3_file_key)
    
    if 'Close' not in data.columns:
        raise ValueError("The data must contain a 'Close' column.")

    # Feature engineering
    features = data[['Close']].values.astype(float)

    # Data scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(features)

    # Prepare the dataset
    X, y = [], []
    for i in range(1, len(dataset)):
        X.append(dataset[i-1:i, 0])
        y.append(dataset[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Predict and revert scaling
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(np.reshape(predicted_price, (predicted_price.shape[0], 1)))

    # Generate signals (1 if price is expected to increase, -1 if expected to decrease)
    y_test_unscaled = scaler.inverse_transform([y_test])
    signals = np.sign(predicted_price - y_test_unscaled.T)
    data = data.iloc[-len(signals):]
    data['Signal'] = signals
    data = data.dropna()

    # Save as CSV to S3
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=True)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=output_csv_file_key)

    # Save as JSON to S3
    json_data = data.to_json(orient='split')
    __save_json_to_s3(json_data, output_json_file_key)
    
    print(f"Data saved as {output_json_file_key} in S3")

def backtest(input_s3_file_key:str='data/ma_output.csv',
                          output_stats=True, output_stats_s3_file_key='data/output_stats.csv',
                          output_plot=True, output_plot_s3_file_key='data/output_plot.html',
                          output_position_record=True, output_position_record_s3_file_key='data/output_position_record.csv',
                          output_return_record=True, output_return_record_s3_file_key='data/output_return_record.csv'):
    """
    Backtest a trading strategy using pre-defined signals and store the backtest results.

    Parameters:
        input_s3_file_key (str, optional): The S3 key of the CSV file containing price and ML-generated signals. 
                                           Default is 'data/ml_output.csv'.
        output_stats (bool, optional): If True, saves the backtest statistics to an S3 bucket. Default is True.
        output_stats_s3_file_key (str, optional): S3 key for saving backtest statistics. Default is 'data/output_stats.csv'.
        output_plot (bool, optional): If True, saves the backtest plot to an S3 bucket. Default is True.
        output_plot_s3_file_key (str, optional): S3 key for saving the backtest plot. Default is 'data/output_plot.html'.
        output_position_record (bool, optional): If True, saves the record of positions during backtesting to an S3 bucket.
                                                 Default is True.
        output_position_record_s3_file_key (str, optional): S3 key for saving position records. 
                                                            Default is 'data/output_position_record.csv'.
        output_return_record (bool, optional): If True, saves the record of returns during backtesting to an S3 bucket. 
                                               Default is True.
        output_return_record_s3_file_key (str, optional): S3 key for saving return records. 
                                                          Default is 'data/output_return_record.csv'.

    Returns:
        pd.DataFrame: A DataFrame containing various statistics and metrics describing the performance of the backtested strategy.

    """
    
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    
    # Get the file content
    file_content = response['Body'].read()
    
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0, parse_dates=True)
    
    # Get trading signals from the 'ML_Signal' column
    trading_signals = df['Signal']
    
    # Use the signals to determine entries (buy) and exits (sell)
    entries = trading_signals == 1
    exits = trading_signals == -1
    
    res = vbt.Portfolio.from_signals(
            df['Close'], 
            entries=entries, 
            exits=exits)
    
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

def visualize_csv_lineplot(input_s3_file_key:str='data/Intraday_Measures.csv',
                           output_s3_file_key:str='data/Intraday_Measures_Output.html'):

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

def __convert_adage3_attribute_json2csv(input_s3_file_key:str='data/Intraday_Measures.json', 
                                     output_s3_file_key:str='data/Intraday_Measures_Output.csv'):

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

def visualise_ADAGE3_json(input_s3_file_key:str='data/Intraday_Measures.json',output_s3_file_key:str='data/Intraday_Measures_Output.html'):
    '''
    Result will be saved in data folder as a html file 
    '''
    __convert_adage3_attribute_json2csv(input_s3_file_key=input_s3_file_key, 
                                     output_s3_file_key='data/temp.csv')
    
    visualize_csv_lineplot(input_s3_file_key='data/temp.csv', 
                         output_s3_file_key=output_s3_file_key)

    return "html file saved in the data folder"





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


def test225(input:str = 'Helwewlo'):
    return input

def list_lib():
  import subprocess
  import json
  data = subprocess.check_output(["pip", "list", "--format", "json"])
  parsed_results = json.loads(data)
  libs = [(element["name"], element["version"]) for 
  element in parsed_results]
  return libs



import json
import pandas as pd
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import openpyxl
import plotly.graph_objs as go
from io import StringIO, BytesIO
import yfinance as yf

# Initialize the S3 client
s3 = boto3.client('s3')
bucket_name = 'unsw-cse-research-slego' 


def __save_text_file_to_s3(text, s3_file_key_output):
    """
    __save a text file to an S3 bucket.

    :param text: Text to be __saved.
    :param s3_file_key_output: S3 key where the text file will be __saved.
    """
    s3.put_object(Body=text, Bucket=bucket_name, Key=s3_file_key_output)

def __save_json_to_s3(data, s3_file_key_output):
    """
    __save a JSON object to an S3 bucket.

    :param data: JSON data to be __saved.
    :param s3_file_key_output: S3 key where the JSON file will be __saved.
    """
    json_data = json.dumps(data)
    s3.put_object(Body=json_data, Bucket=bucket_name, Key=s3_file_key_output)

def __save_xml_to_s3(data, s3_file_key_output):
    """
    __save an XML file to an S3 bucket.

    :param data: XML data to be __saved.
    :param s3_file_key_output: S3 key where the XML file will be __saved.
    """
    s3.put_object(Body=data, Bucket=bucket_name, Key=s3_file_key_output)

def __save_csv_to_s3(data_frame, s3_file_key_output):
    """
    __save a DataFrame as a CSV file to an S3 bucket.

    :param data_frame: DataFrame to be __saved.
    :param s3_file_key_output: S3 key where the CSV file will be __saved.
    """
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=s3_file_key_output)

def __save_parquet_to_s3(data, s3_file_key_output):
    """
    __save a DataFrame as a Parquet file to an S3 bucket.

    :param data: DataFrame to be __saved.
    :param s3_file_key_output: S3 key where the Parquet file will be __saved.
    """
    table = pa.Table.from_pandas(data)
    with BytesIO() as buf:
        pq.write_table(table, buf)
        s3.put_object(Body=buf.getvalue(), Bucket=bucket_name, Key=s3_file_key_output)

def __save_hdf5_to_s3(data, s3_file_key_output):
    """
    __save an HDF5 file to an S3 bucket.

    :param data: Path to the HDF5 file to be __saved.
    :param s3_file_key_output: S3 key where the HDF5 file will be __saved.
    """
    with h5py.File(data, 'r') as f:
        hdf5_data = f['my_dataset'][:]
        hdf5_bytes = hdf5_data.tobytes()
        s3.put_object(Body=hdf5_bytes, Bucket=bucket_name, Key=s3_file_key_output)

def __save_excel_to_s3(data, s3_file_key_output):
    """
    __save an Excel file to an S3 bucket.

    :param data: Path to the Excel file to be __saved.
    :param s3_file_key_output: S3 key where the Excel file will be __saved.
    """
    with open(data, 'rb') as f:
        s3.put_object(Body=f.read(), Bucket=bucket_name, Key=s3_file_key_output)

def __save_html_to_s3(html_content, s3_file_key_output):
    """
    __save HTML content to an S3 bucket.

    :param html_content: HTML content to be __saved.
    :param s3_file_key_output: S3 key where the HTML content will be __saved.
    """
    s3.put_object(Body=html_content, Bucket=bucket_name, Key=s3_file_key_output, ContentType='text/html')

def __save_to_s3(data, file_type, s3_file_key_output):
    """
    __save data to S3 in the specified file format.

    :param data: The data to be __saved.
    :param file_type: The type of file to __save (e.g., 'text', 'json', 'xml', 'csv', 'parquet', 'hdf5', 'excel', 'html').
    :param s3_file_key_output: The S3 key where the file will be __saved.
    """
    if file_type == 'text':
        __save_text_file_to_s3(data, s3_file_key_output)
    elif file_type == 'json':
        __save_json_to_s3(data, s3_file_key_output)
    elif file_type == 'xml':
        __save_xml_to_s3(data, s3_file_key_output)
    elif file_type == 'csv':
        __save_csv_to_s3(data, s3_file_key_output)
    elif file_type == 'parquet':
        __save_parquet_to_s3(data, s3_file_key_output)
    elif file_type == 'hdf5':
        __save_hdf5_to_s3(data, s3_file_key_output)
    elif file_type == 'excel':
        __save_excel_to_s3(data, s3_file_key_output)
    elif file_type == 'html':
        __save_html_to_s3(data, s3_file_key_output)
    else:
        raise ValueError("Unsupported file type provided.")

def get_stock_data_101(ticker: str = 'AAPL', start_date: str = '2020-01-01', end_date: str = '2021-01-01', output_s3_file_key: str = 'data/AAPL_stock_data.csv'):
    """
    Get stock data using yfinance and save to S3 as CSV.

    :param ticker: Stock ticker symbol. Default is 'AAPL' for Apple Inc.
    :param start_date: Start date for fetching data in 'YYYY-MM-DD' format. Default is '2020-01-01'.
    :param end_date: End date for fetching data in 'YYYY-MM-DD' format. Default is '2021-01-01'.
    :param output_s3_file_key: S3 key where the CSV file will be saved. Default is 'data/AAPL_stock_data.csv'.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    __save_to_s3(data, 'csv', output_s3_file_key)
    return data

def preprocess_stock_data_101(input_s3_file_key: str = 'data/AAPL_stock_data.csv', output_s3_file_key: str = 'data/AAPL_stock_data_processed.csv'):
    """
    Preprocess stock data by filling missing values and save processed data back to S3.

    :param input_s3_file_key: S3 key for the input stock data file. Default is 'data/AAPL_stock_data.csv'.
    :param output_s3_file_key: S3 key where the processed data will be saved. Default is 'data/AAPL_stock_data_processed.csv'.
    """
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    data = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')), index_col=0)
    data.fillna(method='ffill', inplace=True)
    __save_to_s3(data, 'csv', output_s3_file_key)
    return data

def compute_simple_moving_average_101(input_s3_file_key: str = 'data/AAPL_stock_data_processed.csv', window_size: int = 20, output_s3_file_key: str = 'data/AAPL_stock_data_SMA.csv'):
    """
    Compute simple moving average for stock data and save updated data back to S3.

    :param input_s3_file_key: S3 key for the input stock data file. Default is 'data/AAPL_stock_data_processed.csv'.
    :param window_size: Window size for the moving average. Default is 20.
    :param output_s3_file_key: S3 key where the updated data will be saved. Default is 'data/AAPL_stock_data_SMA.csv'.
    """
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    data = pd.read_csv(BytesIO(response['Body'].read()), index_col=0)
    data['SMA'] = data['Close'].rolling(window=window_size).mean()
    __save_to_s3(data, 'csv', output_s3_file_key)
    return data

def plot_stock_data_101(input_s3_file_key: str = 'data/AAPL_stock_data_SMA.csv', ticker: str = 'AAPL', output_html_file_key: str = 'plots/AAPL_stock_plot.html'):
    """
    Create a Plotly graph for stock data and save it as an HTML file to S3.

    :param input_s3_file_key: S3 key for the input stock data file. Default is 'data/AAPL_stock_data_SMA.csv'.
    :param ticker: Stock ticker symbol. Default is 'AAPL'.
    :param output_html_file_key: S3 key where the HTML file will be saved. Default is 'plots/AAPL_stock_plot.html'.
    """
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    data = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')), index_col=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', name='Simple Moving Average'))

    fig.update_layout(
        title=f'{ticker} Stock Price and Moving Average',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend'
    )

    # Convert the figure to HTML string and save using the existing function
    html_str = fig.to_html(full_html=True)
    __save_to_s3(html_str, 'html', output_html_file_key)

    return "Plot saved to S3."





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



import json
import requests
import boto3

# Initialize boto3 client



def chat_with_me(input= "Hi my name is Alan/Hirad", ENDPOINT_NAME = "huggingface-pytorch-tgi-inference-2023-09-21-00-28-30-"):
    '''
    Just type anything
    '''
    # Prepare the payload in the format expected by the model
    runtime = boto3.client('runtime.sagemaker', region_name='ap-southeast-2')

    formatted_payload = {
        "inputs": input  # Using 'inputs' as the key, as the error message suggests
    }
    
    payload = json.dumps(formatted_payload)
    
    # Invoke SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=bytes(payload, 'utf-8')
    )
    
    # Parse and return the result
    result = json.loads(response['Body'].read().decode())
    return str(result)





def chat_to_check_sentiment(input="Hi my name is Alan/Hirad and I'm very glad to see you", API_TOKEN = "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"):
    '''
    Just type anything and see the sentiment
    '''
    # Prepare the payload in the format expected by the model

    API_URL="https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    formatted_payload = {
        "inputs": input  # Using 'inputs' as the key, as per Hugging Face's API
    }
    
    payload = json.dumps(formatted_payload)
    
    # Invoke Hugging Face API
    response = requests.post(API_URL, headers=headers, json=formatted_payload)
    
    # Parse and return the result
    result = response.json()
    return str(result)



def check_document_for_sentiment_analysis(API_TOKEN = "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",input_s3_file_key: str = 'data/yfinance_new.csv'):
    from io import BytesIO, StringIO
    import pandas as pd
    '''
    Just select the file and get its sentiment
    '''
    # Prepare the payload in the format expected by the model
    
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    
    # Get the file content
    file_content = response['Body'].read()
    
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0, parse_dates=True)

    API_URL="https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    # Initialize an empty list to store results
    sentiment_results = []

    for index, row in df.iterrows():
        input_text = row['summary']
        
        formatted_payload = {
            "inputs": input_text  # Using 'inputs' as the key, as per Hugging Face's API
        }
        
        payload = json.dumps(formatted_payload)
        
        # Invoke Hugging Face API
        response = requests.post(API_URL, headers=headers, json=formatted_payload)
        
        # Parse and append the result to list
        result = response.json()
        sentiment_results.append(result)
        
    return sentiment_results






