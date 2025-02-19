import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import sent_tokenize
import logging
import pandas as pd
import nltk
import numpy as np
from utils import chunks, InputExample, convert_examples_to_features, softmax
import os

nltk.download('punkt_tab')

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and tokenizer once, globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
model = AutoModelForSequenceClassification.from_pretrained("./model", num_labels=3).to(device)
model.eval()  # Set evaluation mode only once
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def predict(text, batch_size=1):
    sentences = sent_tokenize(text)
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])
    
    for batch in chunks(sentences, batch_size):
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]
        features = convert_examples_to_features(examples, ['positive', 'negative', 'neutral'], 64, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)

        with torch.no_grad():  # Ensures efficient inference
            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            logits = softmax(np.array(logits.cpu()))
            sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
            predictions = np.squeeze(np.argmax(logits, axis=1))
        
        batch_result = pd.DataFrame({
            'sentence': batch,
            'logit': list(logits),
            'prediction': predictions,
            'sentiment_score': sentiment_score
        })
        
        result = pd.concat([result, batch_result], ignore_index=True)
    
    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])

    # Convert numpy types to native Python types before returning
    result['logit'] = result['logit'].apply(lambda x: x.tolist())  # Convert to list
    result['sentiment_score'] = result['sentiment_score'].apply(lambda x: float(x))  # Convert to float

    return {"finbert_prediction": result.prediction.iloc[0], 
            "finbert_sentiment_score": result.sentiment_score.iloc[0]}


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    
    prediction = predict(text)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)