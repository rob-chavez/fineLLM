import argparse
import boto3
import pandas as pd
import io
import requests
import time


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

        batch_size = 1000  # Adjust batch size as needed
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)

        all_embeddings_dfs = []  # Store embeddings DataFrames

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            csv_buffer = io.StringIO()
            batch_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            files = {'file': ('data.csv', csv_buffer, 'text/csv')}
            response = requests.post(args.embedding_url, files=files)

            if response.status_code == 200:
                embeddings_json = response.json().get('embeddings')
                if embeddings_json:
                    embeddings_df = pd.read_json(io.StringIO(embeddings_json), orient='split')
                    all_embeddings_dfs.append(embeddings_df)
                else:
                    print(f"Batch {i+1}: Error: Embeddings not found in response.")
            else:
                print(f"Batch {i+1}: Error: Failed to get embeddings. Status code: {response.status_code}, Response: {response.text}")

            time.sleep(1) #throttle requests.

        if all_embeddings_dfs:
            final_embeddings_df = pd.concat(all_embeddings_dfs)
            upload_to_s3(final_embeddings_df, s3_client, args.bucket_name, args.file_key)
        else:
            print("No embeddings were generated.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()