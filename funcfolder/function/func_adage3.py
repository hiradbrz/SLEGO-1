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
