import pandas as pd
import numpy as np


def preprocess():
    df = pd.read_csv("cook_county_train_val.csv")
    df = df[df['Sale Price'] > 1]
    df = df.drop(columns=['PIN', 'Deed No.', 'Modeling Group'])
    df = df.drop(columns=['Description'])

    return df
