import pandas as pd


def merge_files(file_in, file_list, on_column):
    """
    This function merges multiple CSV files on a specified column.

    Args:
    - file_list: List of paths to the CSV files to merge.
    - on_column: Index Column.

    Returns:
    - Merged DataFrame.
    """
    
    # Initialize an empty DataFrame to store the merged data
    df = pd.read_csv(file_in)
    for f in file_list:
        df_temp = pd.read_csv(f)        
        df_temp.set_index(on_column, inplace=True)
        df = df.join(df_temp)
        
    return df