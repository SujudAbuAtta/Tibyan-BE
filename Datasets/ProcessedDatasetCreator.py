from Datasets.DataPreProcessor import process_dataset
import pandas as pd

dataset_df = pd.read_csv('collective_dataset.csv')
processed_dataset_df = process_dataset(dataset_df)

processed_dataset_df.to_csv("preprocessed_dataset.csv")