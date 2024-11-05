import pandas as pd
import numpy as np


def preprocess():
    df = pd.read_csv("mini2\cook_county_train_val.csv")
    # drop rows with missing sale price
    df = df[df['Sale Price'] > 1]
    # drop categorical variables
    # could regex description for more info
    df = df.drop(columns=['PIN', 'Deed No.', 'Modeling Group', 'Description'])

    # drop empty / mostly empty columns
    # could drop "Multi Property Indicator" as well
    df = df.drop(columns=['Apartments', 'Garage 2 Material', 'Garage 2 Attachment', 'Garage 2 Area',
                          'Number of Commercial Units', 'Use'])

    return df
