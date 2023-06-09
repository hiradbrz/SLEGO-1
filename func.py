#@title
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from jsonschema import validate
import json
import json
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go


def adage3_json_schema_validation(filepath:str= 'jsonfile_quote.json'):
    from jsonschema import validate
    import json
    '''
    input the file name and validate the data
    '''
    f = open(filepath)
    data = json.load(f)

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
    validate(instance=data,schema= schema)
    return 'Valid data'


def convert_adage3_attribute_json2csv(filepath:str='jsonfile_ohlc.json', csv_name:str='json2csv'):
    import json
    import pandas as pd
    from datetime import datetime
    '''
    file will be saved in current repository
    '''


    f = open(filepath)
    data = json.load(f)

    list_att=[]
    list_time=[]

    for i in range(len(data['events'])):
        list_att.append(data['events'][i]['attribute'])
        list_time.append(data['events'][i]['time_object']['timestamp'])


    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(list_att)

    timestamp_strings =list_time
    timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
    #timestamps = [datetime.strptime(ts, timestamp_format) for ts in timestamp_strings]
    try:
        timestamps = [datetime.strptime(ts, timestamp_format) for ts in timestamp_strings]
    except:
        timestamps = timestamp_strings    
    df.index = timestamps

  
    df.to_csv(csv_name+'.csv')

    return df

def visualize_adage3_csv(filepath='data.csv', output_html_name='adage3_data_viz'):
    from pandas.api.types import is_number
    '''
    Result will be saved in current path as a csv file 
    '''
    import plotly.express as px
    import pandas as pd
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go

    df = pd.read_csv(filepath, index_col=0)
    timestamps = df.index
    df['timestamps'] =timestamps
    df.insert(0, 'timestamps', df.pop('timestamps'))
    col_names = list(df.columns)
    coltypes = list(df.dtypes)

    specs =[[{"type": "table"}]]

    for col in col_names:
        specs.append([{"type": "scatter"}])

    fig = make_subplots(
        rows=df.shape[1]+1, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        subplot_titles=['timestamps']+col_names,
        specs=specs)

    fig.add_trace(go.Table(header=dict(values=col_names,font=dict(size=10),align="left"),
            cells=dict(values=[df[k].tolist() for k in df.columns],align = "left")),row=1, col=1)

    for i in range(len(col_names)):
        if coltypes[i]== 'O':
        
            y_out= pd.to_numeric(df[col_names[i]], errors='ignore', downcast=None)
            if not is_number(y_out):
                y_out= df[col_names[i]].apply(len)
            #fig.add_scatter(x=timestamps, y=y_out, hovertemplate=df[col_names[i]],  row=1+i, col=1, name=col_names[i])
            fig.add_trace(go.Scatter(x=timestamps, y=y_out, hovertemplate=df[col_names[i]], name=col_names[i]), row=2+i, col=1)
        else:
            #fig.add_scatter(x=timestamps, y=df[col_names[i]], row=1+i, col=1, name=col_names[i])
            fig.add_trace(go.Scatter(x=timestamps, y=df[col_names[i]],  name=col_names[i]), row=2+i, col=1)

    # Show the plot
    fig.update_layout(height=250*len(col_names), title_text="Visalized ADAGE3 data" ,legend=dict(orientation="h" ))
    fig.write_html(output_html_name+".html")

    return df
    

def visualise_adage3_json(filepath:str='jsonfile_ohlc.json',output_html_name='adage3_data_viz'):
    '''
    Result will be saved in current path as a html file 
    '''

    convert_adage3_attribute_json2csv(filepath=filepath, csv_name='temp' )
    result = visualize_adage3_csv(filepath='temp.csv', output_html_name=output_html_name)

    return result

def validate_and_visualise_adage3_json(filepath:str='jsonfile_ohlc.json', output_html_name:str='adage3_data_viz'):
    '''
    Result will be saved in current path as a html file 
    '''
    
    adage3_json_schema_validation(filepath= filepath)
    convert_adage3_attribute_json2csv(filepath=filepath, csv_name='temp' )
    result = visualize_adage3_csv(filepath='temp.csv', output_html_name=output_html_name)

    return result


def generate_adage3_data_model_template( output_file_path:str='jsonfile_example.json'):
    # Our data model
    import json
    from datetime import datetime
    now= datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    tzs = 'GMT+11'
    tza = 'GMT-5'

    jsonfile_example = {"data_source":"datasource_X", 
                        "dataset_type": "sensor_X", 
                        "dataset_id": "http://bucket-name.s3-website-Region.amazonaws.com", 
                            "time_object":{"timestamp":date_time, "timezone":tzs},
                "events":[{"time_object":{"timestamp":'2019-07-21 13:04:40.3401012', "duration":1, "duration_unit":"second", "timezone":tzs},
                            "event_type":'sensor reading',   
                            "attribute":{"attr1":36.0, "attr2":"abc", "attr3":False}},
                          {"time_object":{"timestamp":'2019-07-22 13:04:40.301022', "duration":1,  "duration_unit":"second","timezone":tzs},
                            "event_type":'sensor reading',   
                            "attribute":{"attr1":37.0, "attr2":"bcd", "attr3":True}},
                          ],  
                }
    # Write pretty print JSON data to file
    with open(output_file_path, "w") as write_file:
        json.dump(jsonfile_example, write_file, indent=4)
    

    return jsonfile_example


def generate_adage3_data_exmaple_yahoo_quote( output_file_path:str='jsonfile_quote.json'):
    # Our data model
    import json
    from datetime import datetime
    now= datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    tzs = 'GMT+11'
    tza = 'GMT-5'
    # get raw data
    import yahoo_fin.stock_info as si
    quote_table = si.get_quote_table("aapl")

    # Our data model
    jsonfile_quote = {"data_source":"yahoo_finance", 
                      "dataset_type": "Daily stock price", 
                      "dataset_id": "http://bucket-name.s3-website-Region.amazonaws.com", 
                      "time_object":{"timestamp":date_time, "timezone":tzs},
                      "events":[{"time_object":{"timestamp":'2019-07-21 13:04:40.3401012', "duration":0, "duration_unit":"day","timezone":tzs},
                            "event_type":'stock quote',   
                            "attribute":quote_table, }],  
                }
    # Write pretty print JSON data to file
    with open(output_file_path, "w") as write_file:
        json.dump(jsonfile_quote, write_file, indent=4)

    return "check folder"



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

def yfinace_ohlc_return_s3data(input_s3_file_key:str='data/SCHW.csv', return_period:int= 1, output_s3_file_key= 'data/SCHW_return.csv'):
    '''
    insert instruction 
    here
    '''
    
    df = __download_s3_file_as_df(input_s3_file_key='data/SCHW.csv')
    # Convert the DataFrame to numeric, turning non-numeric values into NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    df_return = df.pct_change(return_period)
    # df_return = df_return.dropna()
    __save_df_to_s3(df_return, s3_file_key_output=output_s3_file_key)
    return df_return


