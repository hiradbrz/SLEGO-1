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


