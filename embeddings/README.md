Summary of steps to run

1. Launch EC2 instance g5.xlarge w/ AMI UBUNTU with Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
2. chmod 400 [PEM FILE]
3. ssh -i [PEM FILE] ubuntu@[EC2_HOST]
4. install conda via directions here: https://medium.com/@mustafa_kamal/a-step-by-step-guide-to-installing-conda-in-ubuntu-and-creating-an-environment-d4e49a73fc46
5. conda create -n e5 python=3.12 -y
6. conda activate e5
7. $python
8. >>> from huggingface_hub import snapshot_download 
9. >>> snapshot_download(repo_id="intfloat/e5-large-v2", local_dir="./models")
10. >>> exit()
11. conda deactivate
12. git clone https://github.com/rob-chavez/fineLLM.git
13. cd fineLLM/embeddings
14. docker build -t e5 .
15. docker run -p 8000:8000 -t e5
16. run the script --> alternatively to run all companies, use shell file
python embed_headlines.py \
--aws_access_key_id "YOUR KEY" \
--aws_secret_access_key "YOUR KEY" \
--file_key "amazon_inc/_stock_news.csv" \
--bucket_name "bucket_name" \
--embedding_url "embedding_url"