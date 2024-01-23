import json
import requests
import boto3

# Initialize boto3 client



def chat_with_me(input= "Hi my name is Alan/Hirad", ENDPOINT_NAME = "huggingface-pytorch-tgi-inference-2023-09-21-00-28-30-"):
    '''
    Just type anything
    '''
    # Prepare the payload in the format expected by the model
    runtime = boto3.client('runtime.sagemaker', region_name='ap-southeast-2')

    formatted_payload = {
        "inputs": input  # Using 'inputs' as the key, as the error message suggests
    }
    
    payload = json.dumps(formatted_payload)
    
    # Invoke SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=bytes(payload, 'utf-8')
    )
    
    # Parse and return the result
    result = json.loads(response['Body'].read().decode())
    return str(result)





def chat_to_check_sentiment(input="Hi my name is Alan/Hirad and I'm very glad to see you", API_TOKEN = "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"):
    '''
    Just type anything and see the sentiment
    '''
    # Prepare the payload in the format expected by the model

    API_URL="https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    formatted_payload = {
        "inputs": input  # Using 'inputs' as the key, as per Hugging Face's API
    }
    
    payload = json.dumps(formatted_payload)
    
    # Invoke Hugging Face API
    response = requests.post(API_URL, headers=headers, json=formatted_payload)
    
    # Parse and return the result
    result = response.json()
    return str(result)



def check_document_for_sentiment_analysis(API_TOKEN = "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",input_s3_file_key: str = 'data/yfinance_new.csv'):
    from io import BytesIO, StringIO
    import pandas as pd
    '''
    Just select the file and get its sentiment
    '''
    # Prepare the payload in the format expected by the model
    
    # Download the file
    response = s3.get_object(Bucket=bucket_name, Key=input_s3_file_key)
    
    # Get the file content
    file_content = response['Body'].read()
    
    # Load the content into a DataFrame
    df = pd.read_csv(BytesIO(file_content), index_col=0, parse_dates=True)

    API_URL="https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    # Initialize an empty list to store results
    sentiment_results = []

    for index, row in df.iterrows():
        input_text = row['summary']
        
        formatted_payload = {
            "inputs": input_text  # Using 'inputs' as the key, as per Hugging Face's API
        }
        
        payload = json.dumps(formatted_payload)
        
        # Invoke Hugging Face API
        response = requests.post(API_URL, headers=headers, json=formatted_payload)
        
        # Parse and append the result to list
        result = response.json()
        sentiment_results.append(result)
        
    return sentiment_results




