from io import BytesIO

def yfince_ohlc_return_s3data(s3_file_key:str='data/SCHW.csv', return_period:int= 1):

    def download_s3_file_as_df(s3_file_key='data/SCHW.csv'):
        # Download the file
        response = s3.get_object(Bucket=bucket_name, Key=s3_file_key)
        # Get the file content
        file_content = response['Body'].read()
        # Load the content into a DataFrame
        df = pd.read_csv(BytesIO(file_content), index_col=0)
        return df
        
    def save_df_to_s3(df, s3_file_key_output):
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=s3_file_key_output)
    
    df = download_s3_file_as_df(s3_file_key='data/SCHW.csv')
    # Convert the DataFrame to numeric, turning non-numeric values into NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    df_return = df.pct_change(return_period)
    # df_return = df_return.dropna()
    save_file_key = s3_file_key+'_return'+'.csv'
    save_df_to_s3(df_return, s3_file_key_output=save_file_key)
    return df_return

df= yfince_ohlc_return_s3data(s3_file_key='data/SCHW.csv', return_period= 1)
df


#==================================================================================================================
def download_s3_selected_file(s3=s3, bucket_name='unsw-cse-research-slego' , prefix='data', download_path='datafolder', filename='data/temp.csv'):
    # List all objects within a bucket path
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    # Download each file individually
    for object in objects['Contents']:
        if object['Key'] == filename:
            print(f'Downloading {filename}...')
            local_file_path = os.path.join(download_path, filename)
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            s3.download_file(bucket_name, filename, local_file_path)
            return True
    return False

download_s3_selected_file(filename='data/SCHW.csv')