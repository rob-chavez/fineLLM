import boto3
import pandas as pd
import io
import time
import argparse
import math
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# =============================
# ARGUMENT PARSER FOR CONFIGURATION
# =============================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process stock news data from S3 and analyze sentiment using vLLM.")
    parser.add_argument("--aws_access_key_id", type=str, required=True, help="AWS access key ID")
    parser.add_argument("--aws_secret_access_key", type=str, required=True, help="AWS secret access key")
    parser.add_argument("--file_key", type=str, required=True, help="S3 file key (e.g., 'apple_inc/_stock_news.csv')")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    return parser.parse_args()

# =============================
# READ RAW STOCK NEWS DATA FROM S3
# =============================
def read_s3_file(bucket_name, file_key, aws_access_key_id, aws_secret_access_key):
    """Reads a CSV file from an S3 bucket and returns it as a Pandas DataFrame."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-east-1"  
    )

    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df, s3  # Return both the DataFrame and S3 client for later use

# =============================
# SET UP OPENAI CLIENT
# =============================
def setup_openai_client(api_key):
    """Initializes OpenAI client for vLLM API."""
    return OpenAI(
        api_key=api_key,
        base_url="http://54.159.85.60:8000/v1",  # vLLM API server
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1)) 
def get_response(client, headline):
    """Fetches sentiment response from vLLM API."""
    
    prompt = (
        "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive} \n"
        f"Input: '{headline}' \n"
        f"Answer: ")

    response = client.completions.create(model="sql-lora",
                                         prompt=[prompt],
                                         logprobs=True,
                                         temperature=0.6,
                                         max_tokens=1)

    return response

def parse_response(response):
    """Parses and extracts sentiment."""
    try:
        sentiment = response.choices[0].text
        if (sentiment == "") or (sentiment == " ") or (sentiment == None):
            return "error"
        return sentiment.strip().lower()
    except:
        return "error"

def get_confidence(response):
    """Extracts confidence score from logprobs."""
    response_logprobs = (
        response.choices[0].logprobs.token_logprobs[0]
        if hasattr(response.choices[0], "logprobs")
        else 0
    )
    confidence = math.exp(response_logprobs)

    return confidence

# =============================
# PROCESS HEADLINES FUNCTION
# =============================
def process_headlines(df, client):
    """Processes headlines, retrieves sentiment and confidence, and returns a DataFrame."""
    headlines = df["title"].tolist()
    fingpt_sentiments = []
    BATCH_SIZE = 200

    start_time = time.time()

    for idx, headline in enumerate(headlines):
        response = get_response(client, headline)
        sentiment = parse_response(response)
        while (sentiment not in ['neutral', 'positive', 'negative']):
            response = get_response(client, headline)
            sentiment = parse_response(response)
        confidence = get_confidence(response)

        fingpt_sentiments.append({"fingpt_sentiment": sentiment, "fingpt_confidence": confidence})

        if len(fingpt_sentiments) % BATCH_SIZE == 0:
            print(f"Processed {idx + 1} headlines")
            time.sleep(2) 

    elapsed_time = time.time() - start_time
    print(f"Process took {elapsed_time:.2f} seconds to complete.")

    return pd.DataFrame(fingpt_sentiments)

# =============================
# SAVE RESULTS BACK TO S3
# =============================
def upload_to_s3(df, s3_client, bucket_name, file_key):
    """Uploads a DataFrame as a CSV file back to the same S3 bucket and folder."""
    
    # Extract folder path from file_key
    folder_path = "/".join(file_key.split("/")[:-1])  # Get everything except the last part
    output_key = f"{folder_path}/fingpt_sentiments.csv" if folder_path else "fingpt_sentiments.csv"

    # Convert DataFrame to CSV in-memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload to S3
    s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())

    print(f"Results saved back to S3: s3://{bucket_name}/{output_key}")

# =============================
# MAIN EXECUTION
# =============================
if __name__ == "__main__":
    args = parse_arguments()

    bucket_name = "harvard-capstone-bronze-bucket"

    print("Reading stock news data from S3...")
    df, s3_client = read_s3_file(bucket_name, args.file_key, args.aws_access_key_id, args.aws_secret_access_key)

    print("Initializing OpenAI client...")
    client = setup_openai_client(args.openai_api_key)

    print("Processing headlines...")
    final_df = process_headlines(df, client)

    # Save locally
    local_filename = "stock_news_sentiments.csv"
    final_df.to_csv(local_filename, index=False)
    print(f"Results saved locally as '{local_filename}'")

    # Upload back to S3
    upload_to_s3(final_df, s3_client, bucket_name, args.file_key)
