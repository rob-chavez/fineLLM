import boto3
import pandas as pd
import io

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
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df, s3  # Return both the DataFrame and S3 client for later use

# =============================
# SAVE RESULTS BACK TO S3
# =============================
def upload_to_s3(df, s3_client, bucket_name, file_key):
    """Uploads a DataFrame as a CSV file back to the same S3 bucket and folder."""
    
    # Extract folder path from file_key
    folder_path = "/".join(file_key.split("/")[:-1])  # Get everything except the last part
    output_key = f"{folder_path}/senitments_and_embeddings.csv" if folder_path else "senitments_and_embeddings.csv"

    # Convert DataFrame to CSV in-memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload to S3
    s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())

    print(f"Results saved back to S3: s3://{bucket_name}/{output_key}")

#KEYS
aws_access_key_id = "AKIAS74TLZKMHDMOGO5S"
aws_secret_access_key = "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"

# S3 Bucket Name
bucket_name = "harvard-capstone-bronze-bucket"

#COMPANIES
companies = [
    #"flex_ltd",
    #"amazon_inc",
    "apple_inc",
    #"broadcom_inc",
    #"docusign_inc",
    #"dynatrace_inc",
    #"manhattan_associates",
    #"microsoft_corporation",
    "nvidia_corporation",
    #"pure_storage_inc",
]

csv_files = ["_stock_news.csv", "deepseek_sentiments.csv", "finbert_sentiments.csv", "fingpt_sentiments.csv", "embeddings.csv"]

for company in companies:
    file_keys = [f"{company}/{csv_file}" for csv_file in csv_files]

    dfs = {}
    combined_df = pd.DataFrame()

    for file_key in file_keys:
        df, s3_client = read_s3_file(bucket_name, file_key, aws_access_key_id, aws_secret_access_key)
        key = file_key.split("/")[1].replace(".csv", "")
        dfs[key] = df
        print(f"Read {file_key} from S3")
        print(f"Dataframe {key} shape: {df.shape}")

    for key, df in dfs.items():
        print(f"Dataframe {key} shape: {df.shape}")
        if key == "deepseek_sentiments":
            rename_columns = ["deepseek_sentiment", "deepseek_confidence"]
            df.rename(columns=dict(zip(df.columns, rename_columns)), inplace=True)
        if key == "finbert_sentiments":
            rename_columns = ["finbert_sentiment", "finbert_confidence"]
            df.rename(columns=dict(zip(df.columns, rename_columns)), inplace=True)
        if key == "fingpt_sentiments":
            rename_columns = ["fingpt_sentiment", "fingpt_confidence"]
            df.rename(columns=dict(zip(df.columns, rename_columns)), inplace=True)
        combined_df = pd.concat([combined_df, df], axis=1)

    #upload_to_s3(combined_df, s3_client, bucket_name, f"{company}/senitments_and_embeddings.csv")