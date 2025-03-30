import boto3
import pandas as pd
import io
import torch
from datasets import load_dataset, Dataset
import numpy as np

class FinancialSentimentDataLoader:
    def __init__(self, aws_access_key_id, aws_secret_access_key):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = "harvard-capstone-bronze-bucket"
        self.region_name = "us-east-1"
        self.file_key_train = "FinancialPhraseBank/train.csv"
        self.file_key_test = "FinancialPhraseBank/validation.csv"

    def get_financial_phrase_data(self, file_key):
        """
        Fetch Financial PhraseBank data from S3 and process it into a DataFrame.
        """
        # Set up the S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

        # Get the object from S3
        obj = s3.get_object(Bucket=self.bucket_name, Key=file_key)

        # Read the object content into a pandas DataFrame
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        # Rename so that consistent with fingpt-sentiment-data columns
        df.rename(columns={"text": "title", "label": "sentiment"}, inplace=True)

        # Mapping
        mapping = {"positive": 0, "negative": 1, "neutral": 2}
        df["sentiment"] = df["sentiment"].map(mapping).astype(int)

        return df

    def preprocess_fingpt_data(self):
        """
        Load FinGPT dataset, preprocess it, and return train and test DataFrames.
        """
        dataset = load_dataset("FinGPT/fingpt-sentiment-cls")
        train_test_split = dataset["train"].train_test_split(test_size=0.2)
        train_data = train_test_split["train"].to_pandas()
        test_data = train_test_split["test"].to_pandas()

        # Rename the columns for consisitence
        train_data.rename(columns={"input": "title", "output": "sentiment"}, inplace=True)
        test_data.rename(columns={"input": "title", "output": "sentiment"}, inplace=True)

        # Drop the "instruction" column
        train_data.drop(columns=["instruction"], inplace=True)
        test_data.drop(columns=["instruction"], inplace=True)

        # Mapping
        mapping = {"positive": 0, "negative": 1, "neutral": 2}
        train_data["sentiment"] = train_data["sentiment"].map(mapping).astype(int)
        test_data["sentiment"] = test_data["sentiment"].map(mapping).astype(int)

        return train_data, test_data

    def _process_dataset(self, dataframe, tokenizer, text_column="title", label_column="sentiment", max_length=256, num_labels=3):

        labels_list = dataframe[label_column].apply(lambda x: np.array(x) if isinstance(x, list) else np.array([x]))
        
        if num_labels is None:
            num_labels = len(set(np.concatenate(labels_list)))
        
        multi_hot_matrix = np.zeros((len(labels_list), num_labels))
        
        for idx, labels in enumerate(labels_list):
            multi_hot_matrix[idx, labels] = 1  
        
        dataframe["labels"] = multi_hot_matrix.tolist()
        
        def tokenize_function(example):
            return tokenizer(
                example[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        dataset = Dataset.from_pandas(dataframe)

        processed_dataset = dataset.map(tokenize_function, batched=True)
        keep_columns = ["input_ids", "attention_mask", "labels"]
        processed_dataset = processed_dataset.remove_columns([col for col in processed_dataset.column_names if col not in keep_columns])
        processed_dataset.set_format("torch")
        
        return processed_dataset

    def get_training_and_test_data(self, tokenizer):
        """
        Fetch and process Financial PhraseBank and FinGPT datasets, then merge, shuffle, and tokenize them.
        """
        # Load Financial PhraseBank data
        financial_phrase_training_data = self.get_financial_phrase_data(self.file_key_train)
        financial_phrase_test_data = self.get_financial_phrase_data(self.file_key_test)

        # Load FinGPT data
        fingpt_train_data, fingpt_test_data = self.preprocess_fingpt_data()

        # Concatenate train and test datasets
        train = pd.concat([financial_phrase_training_data, fingpt_train_data], axis=0)
        test = pd.concat([financial_phrase_test_data, fingpt_test_data], axis=0)

        # Shuffle the datasets
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        test = test.sample(frac=1, random_state=42).reset_index(drop=True)

        # Convert to Hugging Face datasets and tokenize
        training_data = self._process_dataset(train, tokenizer)
        test_data = self._process_dataset(test, tokenizer)

        return training_data, test_data