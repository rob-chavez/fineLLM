from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import io
import os

app = Flask(__name__)

# Load the model and tokenizer from the local directory
model_path = os.environ.get("TRANSFORMERS_CACHE")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@app.route('/embed', methods=['POST'])
def embed():
    try:
        # Get the CSV file from the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file)

            if 'title' not in df.columns:
                return jsonify({'error': 'DataFrame must contain a column named "title"'})

            titles = df['title'].values
            titles = ["passage: " + title for title in titles]

            # Tokenize the input texts
            batch_dict = tokenizer(titles, max_length=512, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings_np = embeddings.detach().numpy()

            embeddings_df = pd.DataFrame(embeddings_np, index=df.index, columns=[f"dim_{i}" for i in range(embeddings_np.shape[1])])
            result = embeddings_df.to_json(orient='split')

            return jsonify({'embeddings': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')