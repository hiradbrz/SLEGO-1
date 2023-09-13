# %% [markdown]
# # Library

# %%
import requests
import json
import sys
import panel as pn
import pandas as pd
from io import StringIO
import inspect
import time
import param
from collections import ChainMap
import os
import webbrowser
from IPython.display import clear_output
pn.extension(sizing_mode = 'stretch_width')
pn.extension('ace', 'jsoneditor')
import boto3
import ast
import importlib
# Get the directory that the current script is in
try:
    current_directory = os.path.dirname(os.path.realpath(__file__))
except:
    current_directory = os.getcwd()

import datetime

# # Add this directory to the system path
# sys.path.insert(0, current_directory)

# def overwrite_file(filename, content):
#     with open(filename, 'w') as f:
#         f.write(content)

# # Define the content of your new func.py
# content = """
# def test(input:str = 'Hello'):
#     return input
# """
# overwrite_file('func.py', content)

# %% [markdown]
# # Login AWS

# %%
def __login_aws():
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')
    #s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
    folder_name = 'func' 
    return s3, bucket_name, folder_name

s3, bucket_name, folder_name = __login_aws()
s3

# %% [markdown]
# # Define fucntion file

# %%

def delete_files_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# def delete_files_in_folder(folder):
#     pass

#delete_files_in_folder('./funcfolder/function')

# %%

def download_s3_folder(s3=s3, bucket_name='unsw-cse-research-slego' , prefix='function', download_path='funcfolder'):
    # List all objects within a bucket path
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Download each file individually
    for object in objects['Contents']:
        filename = object['Key']
        if filename.endswith("/"): # Don't try to download directories
            continue
        print(f'Downloading {filename}...')
        
        local_file_path = os.path.join(download_path, filename)
        if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path))
        s3.download_file(bucket_name, filename, local_file_path)

# # Use the function:
# download_s3_folder(s3=s3, bucket_name='unsw-cse-research-slego' , prefix='function', download_path='funcfolder')

# %%
# merge python files from a folder
def merge_py_files(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as readfile:
                    outfile.write(readfile.read())
                    outfile.write("\n\n")  # add two newlines between scripts

# merge_py_files('funcfolder./function', 'func.py')
# importlib.reload(func)

# %% [markdown]
# # Function Selector

# %%
import asyncio

class FuncSelector(param.Parameterized):
    def __init__(self, module, datapath='./data/', tag='first', s3=None):
        self.s3=s3
        self.tag=tag
        self.data_format = [pd.DataFrame ] #['_data']
        self.messagebox = pn.widgets.TextInput( name='Message', placeholder='1 select a function, 2 input parameters , 3 click compute button')
        self.datapath = datapath
        self.module = module
        # self.funcs_dict, self.funcs_name_list, self.funcs_argument_list = self.get_function_name() #+type list
        self.btn_compute = pn.widgets.Button(name='Compute Block', button_type = 'primary')
        self.sel_func = pn.widgets.MultiChoice(name='Select analytic services:', options=[''])
        self.update_fuctionlist_ext()
        self.card_param = pn.Card(title='Input parameters:')
        self.sel_func.param.watch(self.select_func_change, 'value') # why not work when i reduce items in select_func_change

        self.btn_compute.on_click(self.btn_compute_clicked)
        self.func_name = None
        self.func_param_dict = None
        self.func = None
        self.data_list = None
        self.func_param_dict = [] #!!
        self.result = None
        self.display = pn.Column()
        self.text_input_save = pn.widgets.TextInput(name='output data names' ,placeholder='result_functionName_[current_time].csv')
        self.funcs_dict_selected = {}
        self.param_type = {}
        self.data_param_list = []
        self.func_desc = pn.widgets.TextAreaInput(name='Instruction' ,height_policy='max')
        self.msgbox_result = pn.widgets.TextAreaInput( name='Result', placeholder='Display result after runiing service', height_policy='max', min_height=250)
        self.paramaccordion = pn.Accordion(pn.widgets.TextInput(name='Parameters will be shown if you select a function', value=''))
        self.paramaccordion.param.watch(self.paramaccordion_change, 'objects')      
        self.dict_textbox_param =  {}
        self.func_json_editor = pn.widgets.TextAreaInput(name='Function editor ',height_policy='max', min_height=250)
        self.btn_update_editor = pn.widgets.Button(name='Update inputs', button_type = 'primary')
        self.btn_update_editor.on_click(self.btn_update_editor_clicked)
        self.is_update = False

    def btn_update_editor_clicked(self,event):
        json_string= self.func_json_editor.value
        json_data = json.loads(json_string)
        function_list = list(json_data.keys())
        self.funcs_dict_selected  = json_data
        self.sel_func.param.unwatch(self.select_func_change)
        self.sel_func.value = function_list
        self.sel_func.param.watch(self.select_func_change, 'value')
        # time.sleep(5)
        for func_name in function_list:
            list_textboxes = self.dict_textbox_param[func_name]
            for textbox in list_textboxes:
                # textbox.param.unwatch(self.text_changed)
                param_name = textbox.name
                textbox.value = json_data[func_name][param_name]
                # textbox.param.watch(self.text_changed,'value')

    def paramaccordion_change(self,event):
        if len(self.paramaccordion.objects)<6:
            self.paramaccordion.active = [len(self.paramaccordion.objects)-1]
            #self.paramaccordion.active = [0]

    def update_fuctionlist_ext(self):
        self.funcs_dict, self.funcs_name_list, self.funcs_argument_list = self.get_function_name()
        # without '__'
        self.funcs_name_list = [x for x in self.funcs_name_list if not x.startswith('__')]
        self.sel_func.options = ['']+self.funcs_name_list

    def btn_compute_clicked(self,event):
        self.msgbox_result.value=''
        self.btn_compute.button_type = 'warning'
        self.display.clear()
        self.messagebox.placeholder = 'Computing...please wait'
        self.msgbox_result.value =''
        try:
            for k,v in self.funcs_dict_selected.items():
                func_name = k.split('^')[0]
                func_param_dict = v
                func = eval('self.module.'+ func_name)
                start_time = time.time()
                self.result = func(**func_param_dict)
                exec_time = time.time() - start_time
                self.msgbox_result.value += '\n'+ '#===Service: ' +func_name+ '===#' +'\n'+str(self.result)
                time.sleep(0.5)
            self.messagebox.placeholder = 'Task finished with time: '+str( round(exec_time,3))+ ' seconds.'
            self.btn_compute.button_type = 'success'
            self.save_result()
        except:
            self.btn_compute.button_type = 'danger'
            self.msgbox_result.value = 'Error: '+str(sys.exc_info()[0])+str(sys.exc_info()[1])

    def save_result(self):
        # Define the JSON data
        data = self.funcs_dict_selected

        # Convert the dictionary to a JSON string and encode it to bytes
        json_data = json.dumps(data).encode('utf-8')

        current_datetime = datetime.datetime.now()
        datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        file_key = 'record/'+datetime_str
        # Upload to S3
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_data, ContentType='application/json')

        return None
    
    def select_func_change(self,event):
        self.func_desc.value = ""
        self.paramaccordion.clear()
        funcs_dict_selected = {}
        self.func_json_editor.value = ''
        functions = self.sel_func.value
        dict_textbox_param = {}
        for function in functions:
        # for loop but skip empty input
            if function == '':
                continue
            func_param_dict = self.funcs_dict[function.split('^')[0]].copy() # must use copy() to avoid change the original dict
            param_inputs = []   
            dict_textbox_param[function] = []
            for k,v in func_param_dict.items():
                arg_input = pn.widgets.LiteralInput(name=k, value=v, tags= [function])
                arg_input.param.watch(self.text_changed,'value')
                param_inputs.append(arg_input)
            self.paramaccordion.append(pn.Column(*param_inputs, name=function))
            
            funcs_dict_selected[function] = func_param_dict 
            dict_textbox_param[function] = param_inputs

            self.func_desc.value += "#==="+ function+"===#\n"
            func = eval('self.module.'+function.split('^')[0]) # only use once for str(func.__doc__) below
            self.func_desc.value += str(func.__doc__)
            self.func_desc.value += "\n"

        if self.sel_func.value == []:
            self.paramaccordion.append(pn.widgets.TextInput(name='Parameters will be shown if you select a function', value=''))

        # update widget text input area in string and json format
        self.func_json_editor.value = json.dumps(funcs_dict_selected, indent=4)
        self.dict_textbox_param = dict_textbox_param
        self.funcs_dict_selected  = funcs_dict_selected

        # self.old_selected_value = self.sel_func.value.copy()
        # self.update_options(event)
        # self.sel_func.param.trigger('options')
        # self.sel_func.value = self.old_selected_value
        

    def text_changed(self,event):
        func_name = event.obj.tags[0]
        param_name = event.obj.name
        self.funcs_dict_selected[func_name][param_name] = event.obj.value
        self.func_json_editor.value = json.dumps(self.funcs_dict_selected, indent=4)

    def get_function_name(self):
        funcs = inspect.getmembers(self.module, inspect.isfunction)
        funcs_name_list = [fun[0] for fun in funcs]
        funcs_argument_list = [inspect.getfullargspec(fun[1])[0]  for fun in funcs]
        #funcs_argument_default_list = [inspect.getfullargspec(fun[1])[3]  for fun in funcs]
        funcs_argument_default_list = [inspect.getfullargspec(fun[1])[3] if inspect.getfullargspec(fun[1])[3] is not None else [] for fun in funcs]

        funcs_dict = [{f:dict(zip(a,d))} for f,a,d in zip(funcs_name_list, funcs_argument_list, funcs_argument_default_list)]
        funcs_dict =dict(ChainMap(*funcs_dict ))
        return funcs_dict, funcs_name_list, funcs_argument_list #+type list

    
    def update_options(self, event):
        # Handle newly selected items
        # keep old selected items
        for item in event.new:
            if item not in event.old:  # Only react to newly selected items
                new_item = f"{item}^"
                self.sel_func.options.append(new_item)
                #self.sel_func.param.trigger('options')  # Manually trigger an update
        
        # use old selection order because iotions.append will change the order
        # Handle deselected items
        for item in event.old:
            if item not in event.new:
                self.sel_func.options = [opt for opt in self.sel_func.options if not opt.endswith('^') or opt in self.sel_func.value]
                #self.sel_func.param.trigger('options')  # Manually trigger an update

        #self.sel_func.param.trigger('options')
        self.msgbox_result.value += 'Selected functions: '+str(self.sel_func.value)+'\n'

    @property
    def view(self):
        return pn.Column(pn.Row( 
            pn.Column(pn.Tabs(('Function selector',self.sel_func), ('Function editor',pn.Column(self.func_json_editor, self.btn_update_editor))),pn.Row(self.messagebox),self.btn_compute,self.msgbox_result),
            self.paramaccordion,
            self.func_desc))
# import func
# importlib.reload(func)
# funcsel = FuncSelector(func)

# funcsel.view.show()

# %% [markdown]
# # File Space

# %%
import logging
import boto3
from botocore.exceptions import ClientError
import webbrowser

class FileSpace(param.Parameterized):
    def __init__(self,s3):
        # Create a session using your AWS credentials
        self.s3 = s3
        self.bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
        self.folder_name = 'data' 
        self.items = []
        self.filemultiselect = pn.widgets.MultiSelect(name='MultiSelect', size=20)
        self.messagebox = pn.widgets.TextAreaInput( name='File details', placeholder='Click on the item for the details.', sizing_mode='stretch_both', height_policy='min', min_height=250)
        # self.filemultiselect.param.watch(self.filemultiselect_change, 'value')
        self.refresh_folder()
        self.button_download = pn.widgets.Button(name='Download', button_type = 'primary')
        self.button_upload = pn.widgets.Button(name='Upload', button_type = 'primary')
        self.button_showdetails = pn.widgets.Button(name='Details', button_type = 'primary')
        self.button_download.on_click(self.download_file)
        self.button_upload.on_click(self.upload_file)
        self.button_showdetails.on_click(self.show_file)
        self.uploder = pn.widgets.FileInput()

        self.btn_update = pn.widgets.Button(name='Update list', button_type = 'primary')
        self.btn_update.on_click(self.update_function)

    def update_function(self,event):
        self.refresh_folder()

    def show_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.filemultiselect.value[-1]
        result = self.s3.get_object(Bucket=self.bucket_name, Key=object_name) 
        text = result["Body"].read().decode()
        self.messagebox.value = text

    def upload_file(self, event):
        try:
            file_bytes = self.uploder.value
            dest_name = self.folder_name+'/'+self.uploder.filename

            # Ensure there's a file to upload
            if not file_bytes or not dest_name:
                self.messagebox.value = "Please select a valid file."
                return

            # Upload to S3
            resp = s3.put_object(Bucket=self.bucket_name, Key=dest_name, Body=file_bytes)

            # Check for a successful upload by examining the response (optional)
            if resp.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
                self.messagebox.value = f'{dest_name} uploaded to {self.bucket_name}/{dest_name} \n'
            else:
                self.messagebox.value = f'Error uploading {dest_name} to S3.'

        except Exception as e:
            # Capture any exception and display an error message
            self.messagebox.value = f"An error occurred: {str(e)}"
        self.refresh_folder()


    def download_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.filemultiselect.value[-1]
        # Generate a presigned URL for the S3 object
        #s3_client = boto3.client('s3')
        try:
            response = self.s3.generate_presigned_url('get_object',
                                                        Params={'Bucket': self.bucket_name,
                                                                'Key': object_name},
                                                        ExpiresIn=3600)
        except ClientError as e:
            logging.error(e)
            return None

        # The response contains the presigned URL
        webbrowser.open(response)

    def refresh_folder(self):   
        self.items =[]
        response = self.s3.list_objects(Bucket=self.bucket_name, Prefix=self.folder_name  )
        for item in response['Contents']:
            self.items.append(item['Key'])
        self.filemultiselect.options = self.items
    
    @property
    def view(self):
        return pn.Column(self.filemultiselect, 
                         pn.Row(self.button_download, self.button_showdetails),
                         pn.Row(self.uploder, self.button_upload),
                         self.btn_update,
                         self.messagebox)

# # test 
# file_space = FileSpace(s3)
# file_space.view.show()


# %% [markdown]
# # Record Space

# %%
import logging
import boto3
from botocore.exceptions import ClientError

class RecordSpace(param.Parameterized):
    def __init__(self,s3):
        # Create a session using your AWS credentials
        self.s3 = s3
        self.bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
        self.folder_name = 'record' 
        self.items = []
        self.filemultiselect = pn.widgets.MultiSelect(name='MultiSelect', size=20)
        self.messagebox = pn.widgets.TextAreaInput( name='File details', placeholder='Click on the item for the details.', sizing_mode='stretch_both', height_policy='min', min_height=250)
        # self.filemultiselect.param.watch(self.filemultiselect_change, 'value')
        self.refresh_folder()
        self.button_download = pn.widgets.Button(name='Download', button_type = 'primary')
        self.button_upload = pn.widgets.Button(name='Upload', button_type = 'primary')
        self.button_showdetails = pn.widgets.Button(name='Details', button_type = 'primary')
        self.button_download.on_click(self.download_file)
        self.button_upload.on_click(self.upload_file)
        self.button_showdetails.on_click(self.show_file)
        self.uploder = pn.widgets.FileInput()

        self.btn_update = pn.widgets.Button(name='Update list', button_type = 'primary')
        self.btn_update.on_click(self.update_function)

    def update_function(self,event):
        self.refresh_folder()

    def show_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.filemultiselect.value[-1]
        result = self.s3.get_object(Bucket=self.bucket_name, Key=object_name) 
        text = result["Body"].read().decode()
        self.messagebox.value = text

    def upload_file(self, event):
        try:
            file_bytes = self.uploder.value
            dest_name = self.folder_name+'/'+self.uploder.filename

            # Ensure there's a file to upload
            if not file_bytes or not dest_name:
                self.messagebox.value = "Please select a valid file."
                return

            # Upload to S3
            resp = s3.put_object(Bucket=self.bucket_name, Key=dest_name, Body=file_bytes)

            # Check for a successful upload by examining the response (optional)
            if resp.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
                self.messagebox.value = f'{dest_name} uploaded to {self.bucket_name}/{dest_name} \n'
            else:
                self.messagebox.value = f'Error uploading {dest_name} to S3.'

        except Exception as e:
            # Capture any exception and display an error message
            self.messagebox.value = f"An error occurred: {str(e)}"
        self.refresh_folder()

    def download_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.filemultiselect.value[-1]
        # Generate a presigned URL for the S3 object
        #s3_client = boto3.client('s3')
        try:
            response = self.s3.generate_presigned_url('get_object',
                                                        Params={'Bucket': self.bucket_name,
                                                                'Key': object_name},
                                                        ExpiresIn=3600)
        except ClientError as e:
            logging.error(e)
            return None

        # The response contains the presigned URL
        webbrowser.open(response)

    def refresh_folder(self):   
        self.items =[]
        response = self.s3.list_objects(Bucket=self.bucket_name, Prefix=self.folder_name  )
        for item in response['Contents']:
            self.items.append(item['Key'])
        self.filemultiselect.options = self.items
    
    @property
    def view(self):
        return pn.Column(self.filemultiselect, 
                         pn.Row(self.button_download, self.button_showdetails),
                         pn.Row(self.uploder, self.button_upload),
                         self.btn_update,
                         self.messagebox)


# %% [markdown]
# # Function Space

# %%
import logging
import boto3
from botocore.exceptions import ClientError

class FuncSpace(param.Parameterized):
    def __init__(self,s3):
        # Create a session using your AWS credentials
        self.s3 = s3
        self.bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
        self.folder_name = 'function' 
        self.items = []
        self.filemultiselect = pn.widgets.MultiSelect(name='MultiSelect', size=20)
        self.messagebox = pn.widgets.TextAreaInput( name='File details', placeholder='Click on the item for the details.', sizing_mode='stretch_both', height_policy='min', min_height=250)
        # self.filemultiselect.param.watch(self.filemultiselect_change, 'value')
        self.refresh_folder()
        self.button_download = pn.widgets.Button(name='Download', button_type = 'primary', disabled = False)
        self.button_upload = pn.widgets.Button(name='Upload', button_type = 'primary')
        self.button_showdetails = pn.widgets.Button(name='Details', button_type = 'primary', disabled = False)
        self.button_download.on_click(self.download_file)
        self.button_upload.on_click(self.upload_file)
        self.button_showdetails.on_click(self.show_file)
        self.uploder = pn.widgets.FileInput()
        self.btn_update = pn.widgets.Button(name='Update list', button_type = 'primary')
        self.btn_update.on_click(self.update_function)

    def update_function(self,event):
        #delete_files_in_folder('./funcfolder/'+self.folder_name )
        download_s3_folder(s3=s3, bucket_name=self.bucket_name, prefix=self.folder_name, download_path='funcfolder')
        merge_py_files('./funcfolder/'+self.folder_name, 'func.py')
        sys.path.insert(0, current_directory)
        importlib.reload(eval('func'))

    def show_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.filemultiselect.value[-1]
        result = self.s3.get_object(Bucket=self.bucket_name, Key=object_name) 
        text = result["Body"].read().decode()
        self.messagebox.value = text

    def upload_file(self, event):
        try:
            file_bytes = self.uploder.value
            dest_name = self.folder_name+'/'+self.uploder.filename

            # Ensure there's a file to upload
            if not file_bytes or not dest_name:
                self.messagebox.value = "Please select a valid file."
                return

            # Upload to S3
            resp = s3.put_object(Bucket=self.bucket_name, Key=dest_name, Body=file_bytes)

            # Check for a successful upload by examining the response (optional)
            if resp.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
                self.messagebox.value = f'{dest_name} uploaded to {self.bucket_name}/{dest_name} \n'
            else:
                self.messagebox.value = f'Error uploading {dest_name} to S3.'

        except Exception as e:
            # Capture any exception and display an error message
            self.messagebox.value = f"An error occurred: {str(e)}"
        self.refresh_folder()
        

    def download_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.filemultiselect.value[-1]
        # Generate a presigned URL for the S3 object
        #s3_client = boto3.client('s3')
        try:
            response = self.s3.generate_presigned_url('get_object',
                                                        Params={'Bucket': self.bucket_name,
                                                                'Key': object_name},
                                                        ExpiresIn=3600)
        except ClientError as e:
            logging.error(e)
            return None

        # The response contains the presigned URL
        webbrowser.open(response)

    def refresh_folder(self):   
        self.items =[]
        response = self.s3.list_objects(Bucket=self.bucket_name, Prefix=self.folder_name  )
        for item in response['Contents']:
            self.items.append(item['Key'])
        self.filemultiselect.options = self.items
    
    @property
    def view(self):
        return pn.Column(self.filemultiselect, 
                         pn.Row(self.button_download, self.button_showdetails),
                         pn.Row(self.uploder, self.button_upload),
                         self.btn_update,
                         self.messagebox)
    


# %% [markdown]
# # Launch application

# %%
import func
import importlib
importlib.reload(func)
import sys

class APP(param.Parameterized):
    def __init__(self, open_in_browser=True):
        self.functionslector = FuncSelector(func,s3)
        self.filespace = FileSpace(s3=s3)
        self.funcspace = FuncSpace(s3=s3)
        self.recordspace = RecordSpace(s3=s3)
        self.tabs = pn.Tabs(('File Space', self.filespace.view ), ('Function Space', self.funcspace.view), ('Record Space', self.recordspace.view))
        self.template = pn.template.MaterialTemplate(title='SLEGO - User config software')
        self.template.main.append(self.functionslector.view)
        self.template.sidebar.append(self.tabs)
        self.server= self.template.show(open=open_in_browser)
        self.funcspace.btn_update.on_click(self.update_function)
        self.update_function(event=None)
        self.functionslector.btn_compute.on_click(self.btn_compute_clicked)
        

    def btn_compute_clicked(self,event):
        self.filespace.refresh_folder()
        self.recordspace.refresh_folder()

    def update_function(self,event):
        self.functionslector.update_fuctionlist_ext()

app = APP()

# %%
app

# %%



