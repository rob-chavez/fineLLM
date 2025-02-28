THESE STEPS SHOULD SET UP A "FINBERT SERVER" -- make sure to change the IP address in finbert_sentiment.py to match public IP of EC2

1. Using EC2 instance: g5.xlarge; using AMI Ubuntu Deep Learning Base
2. $mkdir model
3. huggingface-cli login; you will need your hf creds
4. huggingface-cli download ProsusAI/finbert --local-dir ./model
5. git clone this repo
6. $cd finbert
7. docker build -t finbert .
8. docker run -t --name finbert -p 8000:8000 finbert
