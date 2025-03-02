#!/bin/bash

# List of companies
companies=(
  flex_ltd
  amazon_inc
  apple_inc
  broadcom_inc
  docusign_inc
  dynatrace_inc
  manhattan_associates
  microsoft_corporation
  nvidia_corporation
  pure_storage_inc
)

# AWS and OpenAI keys (replace with your actual keys)
aws_access_key_id=""
aws_secret_access_key=""

# S3 Bucket Name
bucket_name=""

# Embedding App URL
embedding_url="http://3.94.77.86:8000/embed"

# Base file key (suffix)
file_suffix="_stock_news.csv"

# Iterate through each company
for company in "${companies[@]}"; do
  # Construct the full file key
  file_key="${company}/${file_suffix}"

  # Execute the Python script with environment variables
  python embed_headlines.py \
    --aws_access_key_id "$aws_access_key_id" \
    --aws_secret_access_key "$aws_secret_access_key" \
    --file_key "$file_key" \
    --bucket_name "$bucket_name" \
    --embedding_url "$embedding_url"

  echo "Processed: $company"
  sleep 2 
  
done

echo "All companies processed."