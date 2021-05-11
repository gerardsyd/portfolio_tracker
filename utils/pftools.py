import pandas as pd


def csv_to_pkl(csv_file_path: str, pkl_file_path: str):
    df = pd.read_csv(csv_file_path, parse_dates=['Date'], dayfirst=True)
    df.to_pickle(pkl_file_path)
    return df


def pkl_to_csv(csv_file_path: str, pkl_file_path: str, index: bool = False):
    df = pd.read_pickle(pkl_file_path)
    df.to_csv(csv_file_path, index=index)
    return df


if __name__ == "__main__":
    import os
    curr_dir = os.getcwd()

    df = pkl_to_csv(os.path.join(
        curr_dir, 'data/prices_210425.csv'), os.path.join(
        curr_dir, 'data/pf_data.pkl'), True)

    # df = csv_to_pkl('data/pf_data_201015.csv', 'data/pf_trades_CURR.pkl')
    # df = csv_to_pkl('data/test2.csv', 'data/pf_data.pkl')
    # print(df)

    # df = pd.read_pickle('data/pf_trades.pkl')
    # print(df)
    # df.drop(columns=['index'], inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # df.to_pickle('data/pf_trades.pkl')
    # print(df)

    # print(df)
    # print(df[df['Ticker'] == 'AU60RGL00047.FUND'])

    # print(df[df.index == 'TEK.AX'])
    # print(df.query('Name == "NA"'))
    # print(df['Name'] == 'NA')
    # df.to_pickle('data/pf_names.pkl')
    # pass
