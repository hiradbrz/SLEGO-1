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

