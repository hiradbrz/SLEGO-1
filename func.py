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
        "dataset_type_id": {"type": "string"},
        
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

        "required": ["data_source","dataset_type_id","time_object","events"],

        "$defs": {
            "events": {
                "type": "object",
                "properties": {
                    "time_object": {"type": "object",
                                    "properties": {
                                        "timestamp":{"type": "string"},
                                        "duration":{"type": "number"},
                                        "timezone":{"type": "string"},
                                                    },
                    
                                    "required": ["timestamp","duration","timezone"],       
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

def validate(instance, schema, cls=None, *args, **kwargs):
    """
    Validate an instance under the given schema.

        >>> validate([2, 3, 4], {"maxItems": 2})
        Traceback (most recent call last):
            ...
        ValidationError: [2, 3, 4] is too long

    :func:`~jsonschema.validators.validate` will first verify that the
    provided schema is itself valid, since not doing so can lead to less
    obvious error messages and fail in less obvious or consistent ways.

    If you know you have a valid schema already, especially
    if you intend to validate multiple instances with
    the same schema, you likely would prefer using the
    `jsonschema.protocols.Validator.validate` method directly on a
    specific validator (e.g. ``Draft20212Validator.validate``).


    Arguments:

        instance:

            The instance to validate

        schema:

            The schema to validate with

        cls (jsonschema.protocols.Validator):

            The class that will be used to validate the instance.

    If the ``cls`` argument is not provided, two things will happen
    in accordance with the specification. First, if the schema has a
    :kw:`$schema` keyword containing a known meta-schema [#]_ then the
    proper validator will be used. The specification recommends that
    all schemas contain :kw:`$schema` properties for this reason. If no
    :kw:`$schema` property is found, the default validator class is the
    latest released draft.

    Any other provided positional and keyword arguments will be passed
    on when instantiating the ``cls``.

    Raises:

        `jsonschema.exceptions.ValidationError`:

            if the instance is invalid

        `jsonschema.exceptions.SchemaError`:

            if the schema itself is invalid

    .. rubric:: Footnotes
    .. [#] known by a validator registered with
        `jsonschema.validators.validates`
    """
    if cls is None:
        cls = validator_for(schema)

    cls.check_schema(schema)
    validator = cls(schema, *args, **kwargs)
    error = exceptions.best_match(validator.iter_errors(instance))
    if error is not None:
        raise error

def validate_and_visualise_adage3_json(filepath:str='jsonfile_ohlc.json',output_html_name='adage3_data_viz'):
    '''
    Result will be saved in current path as a html file 
    '''
    
    adage3_json_schema_validation(filepath= filepath)
    convert_adage3_attribute_json2csv(filepath=filepath, csv_name='temp' )
    result = visualize_adage3_csv(filepath='temp.csv', output_html_name=output_html_name)

    return result

def visualise_adage3_json(filepath:str='jsonfile_ohlc.json',output_html_name='adage3_data_viz'):
    '''
    Result will be saved in current path as a html file 
    '''

    convert_adage3_attribute_json2csv(filepath=filepath, csv_name='temp' )
    result = visualize_adage3_csv(filepath='temp.csv', output_html_name=output_html_name)

    return result

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

