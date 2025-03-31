import datasets
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np

class SentimentDataLoader:
    def __init__(self):
        self.label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
        self.remap_fpb = {2: 0, 0: 1, 1: 2}

    def get_fpb_data(self, for_training=False):
        fpb = load_dataset("financial_phrasebank", "sentences_50agree")["train"]
        fpb = fpb.train_test_split(test_size=0.2, seed=42)
        fpb = (fpb['train'] if for_training else fpb['test']).to_pandas()
        fpb["label"] = fpb["label"].map(self.remap_fpb)
        fpb.rename(columns={"sentence": "title", "label": "sentiment"}, inplace=True)
        return fpb

    def get_fiqa_data(self, for_training=False):
        dataset = load_dataset('pauri32/fiqa-2018')
        dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        dataset = dataset.train_test_split(test_size=0.2, seed=42) if for_training else dataset.train_test_split(0.226, seed=42)
        dataset = dataset['train' if for_training else 'test'].to_pandas()

        def make_label(x):
            if x < -0.1: return "negative"
            elif -0.1 <= x < 0.1: return "neutral"
            else: return "positive"

        dataset["sentiment"] = dataset.sentiment_score.apply(make_label).map(self.label_mapping)
        dataset.drop(columns=["snippets", "target", "sentiment_score", "aspects", "format", "label"], inplace=True)
        dataset.rename(columns={"sentence": "title"}, inplace=True)
        return dataset

    def get_nwgi_data(self, for_training=False):
        sentiment_map = {
            'strong negative': "negative",
            'moderately negative': "negative",
            'mildly negative': "neutral",
            'strong positive': "positive",
            'moderately positive': "positive",
            'mildly positive': "neutral",
            'neutral': "neutral"
        }
        dataset = datasets.load_dataset('oliverwang15/news_with_gpt_instructions')
        dataset = dataset['train' if for_training else 'test'].to_pandas()
        dataset['sentiment'] = dataset['label'].map(sentiment_map).map(self.label_mapping)
        dataset.rename(columns={"news": "title"}, inplace=True)
        dataset.drop(columns=["prompt", "out", "prompt_tokens", "completion_tokens", "total_tokens", "label"], inplace=True)
        return dataset

    def get_tfns_data(self, for_training=False):
        label_conversion = {0: "negative", 1: "positive", 2: "neutral"}
        dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
        dataset = dataset['train' if for_training else 'validation'].to_pandas()
        dataset['sentiment'] = dataset['label'].map(label_conversion).map(self.label_mapping)
        dataset.rename(columns={"text": "title"}, inplace=True)
        dataset.drop(columns=["label"], inplace=True)
        return dataset

    def process_dataset(self, dataframe, tokenizer, text_column="title", label_column="sentiment", max_length=256, num_labels=3):

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