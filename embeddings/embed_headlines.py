import argparse
import boto3
import pandas as pd
import io
import requests
import json

# =============================
# ARGUMENT PARSER FOR CONFIGURATION
# =============================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process stock news data from S3 and analyze sentiment using vLLM.")
    parser.add_argument("--aws_access_key_id", type=str, required=True, help="AWS access key ID")
    parser.add_argument("--aws_secret_access_key", type=str, required=True, help="AWS secret access key")
    parser.add_argument("--file_key", type=str, required=True, help="S3 file key (e.g., 'apple_inc/_stock_news.csv')")
    parser.add_argument("--bucket_name", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--embedding_url", type=str, required=True, help="URL of the embedding app (e.g., http://localhost:5000/embed)")
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
# SAVE RESULTS BACK TO S3
# =============================
def upload_to_s3(df, s3_client, bucket_name, file_key):
    """Uploads a DataFrame as a CSV file back to the same S3 bucket and folder."""

    # Extract folder path from file_key
    folder_path = "/".join(file_key.split("/")[:-1])  # Get everything except the last part
    output_key = f"{folder_path}/embeddings.csv" if folder_path else "embeddings.csv"

    # Convert DataFrame to CSV in-memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload to S3
    s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())

    print(f"Results saved back to S3: s3://{bucket_name}/{output_key}")

# =============================
# MAIN FUNCTION
# =============================
def main():
    args = parse_arguments()

    try:
        df, s3_client = read_s3_file(args.bucket_name, args.file_key, args.aws_access_key_id, args.aws_secret_access_key)

        # Send CSV to embedding app
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0) #reset buffer position

        files = {'file': ('data.csv', csv_buffer, 'text/csv')}
        response = requests.post(args.embedding_url, files=files)

        if response.status_code == 200:
            embeddings_json = response.json().get('embeddings')
            if embeddings_json:
                embeddings_df = pd.read_json(embeddings_json, orient='split')
                upload_to_s3(embeddings_df, s3_client, args.bucket_name, args.file_key)
            else:
                print("Error: Embeddings not found in response.")
        else:
            print(f"Error: Failed to get embeddings. Status code: {response.status_code}, Response: {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()