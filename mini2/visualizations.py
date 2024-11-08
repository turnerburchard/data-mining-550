import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import preprocessing

df = preprocessing.preprocess()

def scatter_plot_x_y(x,y):
    print(f"\'{x}\' vs \'{y}\'")
    plt.scatter(np.log(df[x]), np.log(df[y]))
    plt.show()

for var in ['Building Square Feet','Lot Size','Estimate (Land)','Estimate (Building)','Age','Age Decade']:
    scatter_plot_x_y(var, 'Sale Price')
    