# bedrock_llm.py
import os
from langchain_aws import ChatBedrock

def get_bedrock_llm(model_id=None):
    """
    Initialize a Bedrock LLM client.
    
    Args:
        model_id (str): The model ID to use. If None, defaults to Claude 3 Haiku.
        
    Returns:
        langchain_aws.Bedrock: An initialized Bedrock LLM client
    """
    # Default model if none provided
    if model_id is None:
        model_id = "us.meta.llama3-3-70b-instruct-v1:0"
    
    # Configure Bedrock client
    bedrock_runtime_args = {
        "region_name": os.getenv("AWS_REGION", "us-east-1"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }
    
    # Model parameters
    model_kwargs = {
        "temperature": 0.7,
        "max_tokens": 8000,
        "top_p": 0.9,
    }
    
    # Initialize the Bedrock client
    llm = ChatBedrock(
        model_id=model_id,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        credentials_profile_name=None,  # Use environment variables for auth
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 4096,  # Reduced from the default, to be safe
            "top_p": 0.9,
        }
    )
    
    return llm