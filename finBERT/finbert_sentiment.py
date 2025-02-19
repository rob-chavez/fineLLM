import boto3
import pandas as pd
import io
import argparse
import requests
import time
import json

# =============================
# ARGUMENT PARSER FOR CONFIGURATION
# =============================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process stock news data from S3 and analyze sentiment using vLLM.")
    parser.add_argument("--aws_access_key_id", type=str, required=True, help="AWS access key ID")
    parser.add_argument("--aws_secret_access_key", type=str, required=True, help="AWS secret access key")
    parser.add_argument("--file_key", type=str, required=True, help="S3 file key (e.g., 'apple_inc/_stock_news.csv')")
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
# PREDICT VIA FINBERT FLASK SERVER
# =============================
def predict(text):
    url = "http://100.28.231.40:8000/predict"  #make sure url is correct given spot instance
    response = requests.post(url, json={"text": text})
    if response.status_code == 200:
        return response.json()  
    else:
        return {"error": f"Failed with status code {response.status_code}"}

# =============================
# PROCESS HEADLINES FUNCTION
# =============================
def process_headlines(df):
    """Processes headlines, retrieves sentiment and confidence, and returns a DataFrame."""
    headlines = df["title"].tolist()
    sentiments = []
    BATCH_SIZE = 200

    start_time = time.time()

    for idx, headline in enumerate(headlines):
        print(headline, end=" ")
        sentiments.append(predict(headline))
        if len(sentiments) % BATCH_SIZE == 0:
            print(f"Processed {idx + 1} headlines")
            time.sleep(2)  # Throttling to avoid hitting rate limits

    elapsed_time = time.time() - start_time
    print(f"Process took {elapsed_time:.2f} seconds to complete.")

    return pd.DataFrame(sentiments)

# Main execution of the script
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Read stock news data from S3
    print("Reading stock news data from S3...")
    df, s3_client = read_s3_file(
        bucket_name="harvard-capstone-bronze-bucket", 
        file_key=args.file_key, 
        aws_access_key_id=args.aws_access_key_id, 
        aws_secret_access_key=args.aws_secret_access_key
    )

    # Process the headlines and get the results
    print("Processing headlines for sentiment analysis...")
    df2 = process_headlines(df)

    # Optionally, save the results to a new CSV
    df2.to_csv("sentiment_analysis_results.csv", index=False)
    print("Sentiment analysis results saved to 'sentiment_analysis_results.csv'.")
