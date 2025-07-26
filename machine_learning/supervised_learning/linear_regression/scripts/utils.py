import pandas as pd

def load_house_price_data(path:str, as_dataframe:bool=False):
  df = pd.read_csv(path)
  if as_dataframe:
    X = df[['size']]
    y = df['price']
    return X, y
  else:
    return df['size'].tolist(), df['price'].tolist()
