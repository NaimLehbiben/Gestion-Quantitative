import pandas as pd
from matplotlib import pyplot as plt

panel_data_df = pd.read_pickle('panel_data.pkl')
print(panel_data_df.shape)
