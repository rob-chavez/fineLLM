import torch
from sklearn.metrics import accuracy_score, f1_score

class SentimentInference:
    def __init__(self, inference_model, tokenizer):
        self.inference_model, self.device = self.prepare_inference_model(inference_model)
        self.tokenizer = tokenizer

    @staticmethod
    def prepare_inference_model(inference_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_model.to(device)
        return inference_model, device

    def perform_inference(self, input_text):

        # Tokenize the input text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,  # Adjust max length if necessary
        )

        # Move inputs to the device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.inference_model(**inputs)

        # Extract logits and predictions
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        # Map predictions to sentiment labels
        #label_mapping = self.inference_model.config.id2label
        #predicted_label = label_mapping[predicted_class]
        #print(predicted_label)

        return predicted_class

    def process_dataframe(self, dataframe):

        predictions = []

        # Perform inference for each title in the DataFrame
        for title in dataframe["title"]:
            predictions.append(self.perform_inference(title))

        # Add predictions to the DataFrame
        dataframe["prediction"] = predictions

        # Compute evaluation metrics
        true_labels = dataframe["sentiment"].tolist()
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="macro")

        return dataframe, accuracy, f1
