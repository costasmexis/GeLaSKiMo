import pandas as pd
import requests
import scipy.io
import io

def get_data_zenodo(url: str, filename: str, key: str) -> pd.DataFrame:
    """
    Get data from Zenodo and save it as a csv file.
    """
    response = requests.get(url)
    
    if response.status_code == 200:
        content_bytes = io.BytesIO(response.content)
        data = scipy.io.loadmat(content_bytes)
        
        try:
            df = pd.DataFrame(data[key].T)
            return df
        except ValueError:
            m = data[key]
            con_list = [[element for element in upperElement] for upperElement in m]
            allEnzymes = con_list[0][0][4][0][0]
            allEnzymes = [i[0][0] for i in allEnzymes]
            
            s = con_list[0][0][1][0][0]
            mcc = pd.DataFrame(s.reshape(200000,86))
            mcc.columns = allEnzymes
            return mcc
    else:
        print("Failed to download the file.")

def get_dataset(data: pd.DataFrame, labels: pd.DataFrame, names: pd.DataFrame):
    names = [_[0] for _ in names.iloc[0].values]
    labels = [_[0] for _ in labels.iloc[0].values]
    data.columns = names
    data['stability'] = labels
    data['stability'] = data['stability'].map({'s': 1, 'ns': 0}).astype(int)
    return data
