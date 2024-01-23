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



