#!/bin/bash

# List of companies
companies=(
  amazon_inc
  apple_inc
  broadcom_inc
  docusign_inc
  dynatrace_inc
  flex_ltd
  manhattan_associates
  microsoft_corporation
  nvidia_corporation
  pure_storage_inc
)

# AWS and OpenAI keys (replace with your actual keys)
aws_access_key_id="AWS_ACCESS_KEY_ID"
aws_secret_access_key="AWS_SECRET_ACCESS_KEY"

# S3 Bucket Name
bucket_name="BUCKET_NAME"

# Embedding App URL
embedding_url="http://EMBEDDING_URL:8000/embed"

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
    --embedding_app_url "$embedding_url"

  echo "Processed: $company"
  sleep 2 
  
done

echo "All companies processed."