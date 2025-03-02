Summary of steps to run

1. Launch EC2 instance g5.xlarge w/ AMI UBUNTU with Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
2. chmod 400 [PEM FILE]
3. ssh -i [PEM FILE] ubuntu@[EC2_HOST]
4. install conda via directions here: https://medium.com/@mustafa_kamal/a-step-by-step-guide-to-installing-conda-in-ubuntu-and-creating-an-environment-d4e49a73fc46
5. conda create -n vllm python=3.12 -y
6. conda activate vllm
7. pip install vllm 
8. vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --enable-prefix-caching

9. run the script 
python deepseek_sentiment.py \
--aws_access_key_id "YOUR KEY" \
--aws_secret_access_key "YOUR KEY" \
--file_key "amazon_inc/_stock_news.csv" \
--openai_api_key "EMPTY"