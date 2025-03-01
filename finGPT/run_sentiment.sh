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
aws_access_key_id="YOUR KEY"
aws_secret_access_key="YOUR KEY"
openai_api_key="EMPTY"

# Base file key (suffix)
file_suffix="_stock_news.csv"

# Iterate through each company
for company in "${companies[@]}"; do
  # Construct the full file key
  file_key="${company}/${file_suffix}"

  # Execute the Python script
  python fingpt_sentiment.py \
    --aws_access_key_id "$aws_access_key_id" \
    --aws_secret_access_key "$aws_secret_access_key" \
    --file_key "$file_key" \
    --openai_api_key "$openai_api_key"

  # Optional: Add a delay or message between runs
  echo "Processed: $company"
  #sleep 2 #optional sleep for 2 seconds between runs.
done

echo "All companies processed."