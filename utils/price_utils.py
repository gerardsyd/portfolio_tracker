import pandas as pd


def calc_avg_price(self, df: pd.DataFrame) -> pd.DataFrame:
    df['grouping'] = df['CumQuan'].eq(0).shift().cumsum().fillna(
        0.)  # create group for each group of shares bought / sold
    DF = df.groupby('grouping', as_index=False).apply(
        lambda x: x.CFBuy.sum()/x.QBuy.sum()).reset_index(drop=True)
    DF.columns = ['grouping', 'AvgCostAdj']
    df = pd.merge(df, DF, how='left', on='grouping')
    return df
