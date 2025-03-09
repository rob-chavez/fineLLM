import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class DataProcessor:
    def __init__(self, snes: pd.DataFrame, sd: pd.DataFrame):
        self.snes = snes.copy()  
        self.sd = sd.copy()      
        self.embeddings_columns = [f"dim_{i}" for i in range(1024)]
        self.sentiment_columns = [
            "vader_score_title", "deepseek_sentiment", "deepseek_confidence",
            "finbert_sentiment", "finbert_confidence", "fingpt_sentiment", "fingpt_confidence"
        ]
        self.most_relevant_features = ["open", "high", "low", "close", "volume"]

    def preprocess_data(self):
        # Preprocess `snes`
        cols_to_keep = ["date"] + self.embeddings_columns + self.sentiment_columns
        self.snes = self.snes.loc[:, cols_to_keep]
        self.snes.loc[:, "date"] = pd.to_datetime(self.snes["date"], format="%Y-%m-%d")
        self.snes.set_index("date", inplace=True)
        self.snes = self.snes.infer_objects()  # Explicitly infer objects to remove ambiguity
        self.snes.sort_index(inplace=True)

        # Preprocess `sd`
        cols_to_keep = ["date"] + self.most_relevant_features
        self.sd = self.sd.loc[:, cols_to_keep]
        self.sd.loc[:, "date"] = pd.to_datetime(self.sd["date"], format="%Y-%m-%d")
        self.sd.set_index("date", inplace=True)
        self.sd = self.sd.infer_objects()  # Explicitly infer objects to remove ambiguity
        self.sd.sort_index(inplace=True)

    def sentiment_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
        for sentiment_column in ["deepseek_sentiment", "finbert_sentiment", "fingpt_sentiment"]:
            df.loc[:, sentiment_column] = df[sentiment_column].map(sentiment_mapping).fillna(0)
        return df

    def calculate_slopes(self, X: np.ndarray, y: np.ndarray) -> float:
        if X.shape[0] == 1:
            return y[0]
        model = LinearRegression()
        model.fit(X, y)
        return float(model.coef_[0])

    def process(self):
        # Merge and clean data
        merged_df = self.sd.merge(self.snes, how="outer", left_index=True, right_index=True)
        merged_df_filled = merged_df.fillna(0).infer_objects(copy=False) 

        merged_df_filled = self.sentiment_to_numeric(merged_df_filled)

        # Generate 'averaged' DataFrame
        averaged = merged_df_filled.groupby(merged_df_filled.index).mean()

        # Filter rows where confidence columns are non-zero
        filtered_df = merged_df_filled[
            (merged_df_filled[["deepseek_confidence", "finbert_confidence", "fingpt_confidence"]].ne(0)).all(axis=1)
        ]
        embeddings_df = filtered_df[self.embeddings_columns]
        record_counts = embeddings_df.groupby(embeddings_df.index.date).size()

        # Generate 'slope' DataFrame
        slopes_df = pd.DataFrame()
        for idx, d in enumerate(list(record_counts.index)):
            slopes = []
            X = np.array(range(0, record_counts.iloc[idx])).reshape(-1, 1)
            for dim in embeddings_df.columns:
                y = np.array(embeddings_df[embeddings_df.index.date == d][dim].values)
                slopes.append(self.calculate_slopes(X, y))

            dates_slopes = pd.DataFrame(slopes).T
            slopes_df = pd.concat([slopes_df, dates_slopes], axis=0)
        slopes_df = slopes_df.add_prefix("dim_")

        # Combine sentiments and slopes
        sentiments_df = filtered_df[self.sentiment_columns]
        avg_df = sentiments_df.groupby(sentiments_df.index).mean()
        slopes_df.index = avg_df.index
        slope = slopes_df.merge(avg_df, left_index=True, right_index=True, how="inner")
        slope = self.sd.merge(slope, how="outer", left_index=True, right_index=True)
        slope = slope.fillna(0).infer_objects(copy=False)  # Explicitly handle downcasting

        return averaged, slope