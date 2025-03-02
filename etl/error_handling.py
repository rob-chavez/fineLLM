import boto3
import pandas as pd
import io
import json
import time
import argparse
from openai import OpenAI
from llm_confidence.logprobs_handler import LogprobsHandler
from tenacity import retry, stop_after_attempt, wait_fixed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fix sentiment errors in stock news data.")
    parser.add_argument("--aws_access_key_id", type=str, required=True)
    parser.add_argument("--aws_secret_access_key", type=str, required=True)
    parser.add_argument("--file_key", type=str, required=True)
    parser.add_argument("--openai_api_key", type=str, required=True)
    return parser.parse_args()

def read_s3_file(bucket_name, file_key, aws_access_key_id, aws_secret_access_key):
    s3 = boto3.client(
        's3', aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key, region_name="us-east-1"
    )
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df, s3

def setup_openai_client(api_key):
    return OpenAI(api_key=api_key, base_url="http://54.221.175.144:8000/v1")

SYS_PROMPT = (
    "You are an expert stock trader who provides sentiment analysis of news headlines. "
    "Return a JSON object with 'sentiment' as 'positive', 'negative', or 'neutral'."
)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_response(client, headline):
    prompt = f"Is this news headline positive, negative, or neutral: {headline}. "
    "Provide only a JSON object with 'sentiment' as the key."
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        logprobs=True, temperature=0.6, max_tokens=512, response_format={'type': 'json_object'}
    )
    return response

def parse_response(response):
    try:
        message = json.loads(response.choices[0].message.content)
        return message.get("sentiment", "error")
    except:
        return "error"

def get_confidence(response):
    try:
        logprobs_handler = LogprobsHandler()
        response_logprobs = response.choices[0].logprobs.content if hasattr(response.choices[0], "logprobs") else []
        logprobs_formatted = logprobs_handler.format_logprobs(response_logprobs)
        confidence = logprobs_handler.process_logprobs(logprobs_formatted)
        return confidence.get("sentiment", 1)
    except:
        return 1

def process_errors(df, client):
    while "error" in df["sentiment"].values:
        error_df = df[df["sentiment"] == "error"].copy()
        for idx, row in error_df.iterrows():
            response = get_response(client, row["title"])
            df.at[idx, "sentiment"] = parse_response(response)
            df.at[idx, "confidence"] = get_confidence(response)
        print("Errors processed. Checking again...")
    return df

def upload_to_s3(df, s3_client, bucket_name, file_key):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
    print(f"Updated file saved to S3: s3://{bucket_name}{file_key}")

if __name__ == "__main__":
    args = parse_arguments()
    bucket_name = "harvard-capstone-bronze-bucket"
    df, s3_client = read_s3_file(bucket_name, args.file_key, args.aws_access_key_id, args.aws_secret_access_key)
    client = setup_openai_client(args.openai_api_key)
    df = process_errors(df, client)
    upload_to_s3(df, s3_client, bucket_name, args.file_key)