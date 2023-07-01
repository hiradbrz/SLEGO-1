
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
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# %% [markdown]
# # Login AWS

# %%
def __login_aws(aws_access_key_id:str=ACCESS_KEY,
                aws_secret_access_key:str=SECRET_KEY):
    # Create a session using your AWS credentials
    s3 = boto3.client('s3', 
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)
    bucket_name = 'unsw-cse-research-slego'  # replace with your bucket name
    folder_name = 'func' 
    return s3, bucket_name, folder_name

s3, bucket_name, folder_name = __login_aws()

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

delete_files_in_folder('./funcfolder/function')

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
        self.sel_func = pn.widgets.MultiChoice(name='Select an analytic service:', options=[''])
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
        if len(self.paramaccordion.objects)>0:
            self.paramaccordion.active = [len(self.paramaccordion.objects)-1]

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
        for k,v in self.funcs_dict_selected.items():
            func_name = k
            func_param_dict = v
            func = eval('self.module.'+ func_name)
            start_time = time.time()
            self.result = func(**func_param_dict)
            exec_time = time.time() - start_time
            self.msgbox_result.value += '\n'+ '#===Service: ' +func_name+ '===#' +'\n'+str(self.result)
            time.sleep(0.5)
        self.messagebox.placeholder = 'Task finished with time: '+str( round(exec_time,3))+ ' seconds.'
        self.btn_compute.button_type = 'success'
    
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
            func = eval('self.module.'+function)
            func_param_dict = self.funcs_dict[function]
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
            self.func_desc.value += str(func.__doc__)
            self.func_desc.value += "\n"

        if self.sel_func.value == []:
            self.paramaccordion.append(pn.widgets.TextInput(name='Parameters will be shown if you select a function', value=''))

        # update widget text input area in string and json format
        self.func_json_editor.value = json.dumps(funcs_dict_selected, indent=4)
        self.dict_textbox_param = dict_textbox_param
        self.funcs_dict_selected  = funcs_dict_selected
        
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

        self.btn_update = pn.widgets.Button(name='Update datasets', button_type = 'primary')
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

    def upload_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.uploder.value
        # Generate a presigned URL for the S3 object
        #s3_client = boto3.client('s3')
        try:
            response = self.s3.generate_presigned_url('put_object',
                                                        Params={'Bucket': self.bucket_name,
                                                                'Key': object_name},
                                                        ExpiresIn=3600)
        except ClientError as e:
            logging.error(e)
            return None
        # The response contains the presigned URL
        webbrowser.open(response)

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
        self.button_download = pn.widgets.Button(name='Download', button_type = 'primary', disabled = True)
        self.button_upload = pn.widgets.Button(name='Upload', button_type = 'primary')
        self.button_showdetails = pn.widgets.Button(name='Details', button_type = 'primary', disabled = True)
        self.button_download.on_click(self.download_file)
        self.button_upload.on_click(self.upload_file)
        self.button_showdetails.on_click(self.show_file)
        self.uploder = pn.widgets.FileInput()
        self.btn_update = pn.widgets.Button(name='Update function', button_type = 'primary')
        self.btn_update.on_click(self.update_function)

    def update_function(self,event):
        delete_files_in_folder('./funcfolder/'+self.folder_name )
        download_s3_folder(s3=s3, bucket_name=self.bucket_name, prefix=self.folder_name, download_path='funcfolder')
        merge_py_files('./funcfolder/'+self.folder_name, 'func.py')
        importlib.reload(func)

    def show_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.filemultiselect.value[-1]
        result = self.s3.get_object(Bucket=self.bucket_name, Key=object_name) 
        text = result["Body"].read().decode()
        self.messagebox.value = text

    def upload_file(self,event):
        if self.filemultiselect.value == []:
            return
        object_name = self.uploder.value
        # Generate a presigned URL for the S3 object
        #s3_client = boto3.client('s3')
        try:
            response = self.s3.generate_presigned_url('put_object',
                                                        Params={'Bucket': self.bucket_name,
                                                                'Key': object_name},
                                                        ExpiresIn=3600)
        except ClientError as e:
            logging.error(e)
            return None
        # The response contains the presigned URL
        webbrowser.open(response)

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
        self.tabs = pn.Tabs(('File Space', self.filespace.view ), ('Function Space', self.funcspace.view))
        self.template = pn.template.MaterialTemplate(title='SLEGO - User config software')
        self.template.main.append(self.functionslector.view)
        self.template.sidebar.append(self.tabs)
        #self.server= self.template.show(open=open_in_browser)
        self.funcspace.btn_update.on_click(self.update_function)
        self.update_function(event=None)
        self.functionslector.btn_compute.on_click(self.btn_compute_clicked)

    def btn_compute_clicked(self,event):
        self.filespace.refresh_folder()

    def update_function(self,event):
        self.functionslector.update_fuctionlist_ext()

app = APP()
app.template.servable()

