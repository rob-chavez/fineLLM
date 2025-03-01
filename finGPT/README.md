Summary of steps to run

1. Launch EC2 instance g5.xlarge w/ AMI UBUNTU with Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
2. chmod 400 [PEM FILE]
3. ssh -i [PEM FILE] ubuntu@[EC2_HOST]
4. install conda via directions here: https://medium.com/@mustafa_kamal/a-step-by-step-guide-to-installing-conda-in-ubuntu-and-creating-an-environment-d4e49a73fc46
5. conda create -n vllm python=3.12 -y
6. conda activate vllm
7. pip install vllm 
8. python
9. download model into .cache/huggingface/hub/ using python command below:
$python
>>> from huggingface_hub import snapshot_download
>>> sql_lora_path = snapshot_download(repo_id="FinGPT/fingpt-mt_llama3-8b_lora")
>>> exit()
10. huggingface-cli login #you need to login and agree to  meta-llama terms on HF
11. vllm serve meta-llama/Meta-Llama-3-8B \
    --enable-lora \
    --lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--FinGPT--fingpt-mt_llama3-8b_lora/snapshots/5b5850574ec13e4ce7c102e24f763205992711b7/

12 run the script 
python fingpt_sentiment.py \
--aws_access_key_id "YOUR KEY" \
--aws_secret_access_key "YOUR KEY" \
--file_key "amazon_inc/_stock_news.csv" \
--openai_api_key "EMPTY"