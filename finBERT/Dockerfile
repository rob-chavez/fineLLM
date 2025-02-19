FROM python:3.9

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY finbert.py ./finbert.py
COPY utils.py ./utils.py
COPY model ./model
#make sure to download the weights from huggingface or COPY model will not work
#huggingface-cli login
#huggingface-cli download ProsusAI/finbert --local-dir ./model
EXPOSE 8000
CMD ["python", "finbert.py"]
