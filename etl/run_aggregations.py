from dataprocessor import DataProcessor
import boto3
import pandas as pd
import io
import argparse


# =============================
# ARGUMENT PARSER FOR CONFIGURATION
# =============================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process stock data and sentiment + embeddings data.")
    parser.add_argument("--aws_access_key_id", type=str, required=True, help="AWS access key ID")
    parser.add_argument("--aws_secret_access_key", type=str, required=True, help="AWS secret access key")
    parser.add_argument("--csv_file_one", type=str, required=True, help="S3 file key (e.g., 'apple_inc/senitments_and_embeddings.csv')")
    parser.add_argument("--csv_file_two", type=str, required=True, help="S3 file key (e.g., 'apple_inc/_stock_data.csv')")
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
def upload_to_s3(df, s3_client, bucket_name, file_key, new_file):
    """Uploads a DataFrame as a CSV file back to the same S3 bucket and folder."""
    
    # Extract folder path from file_key
    folder_path = "/".join(file_key.split("/")[:-1])  # Get everything except the last part
    add_folder = "aggregations_version_1"
    output_key = f"{folder_path}/{add_folder}/{new_file}" if folder_path else new_file

    # Convert DataFrame to CSV in-memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)

    # Upload to S3
    s3_client.put_object(Bucket="harvard-capstone-silver-bucket", Key=output_key, Body=csv_buffer.getvalue())
    print(f"Results saved back to S3: s3://{bucket_name}/{output_key}")

if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_arguments()
    bucket_name="harvard-capstone-bronze-bucket"

    # Read sentiment and embedding data from S3
    print("Reading sentiment and embedding data from S3...")
    sentiment_and_embeddings_data, s3_client_one = read_s3_file( 
        bucket_name=bucket_name,
        file_key=args.csv_file_one, 
        aws_access_key_id=args.aws_access_key_id, 
        aws_secret_access_key=args.aws_secret_access_key
    )

    # Read stock data from S3
    print("Reading stock data from S3...")
    stock_data, s3_client_two = read_s3_file( 
        bucket_name=bucket_name,
        file_key=args.csv_file_two, 
        aws_access_key_id=args.aws_access_key_id, 
        aws_secret_access_key=args.aws_secret_access_key
    )

    # Aggregate my sentiment and embeddings data and stock data
    processor = DataProcessor(sentiment_and_embeddings_data, stock_data)
    processor.preprocess_data()
    averaged, slope = processor.process()
    
    #  upload back to S3
    upload_to_s3(averaged, s3_client_one, bucket_name, args.csv_file_one, "aggregated_by_average.csv")
    upload_to_s3(slope, s3_client_two, bucket_name, args.csv_file_one, "aggregated_by_slope.csv")