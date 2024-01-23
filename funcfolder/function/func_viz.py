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
