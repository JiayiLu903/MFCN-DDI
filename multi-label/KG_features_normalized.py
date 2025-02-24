import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


input_file = "drug_KG_features.csv"
output_file = "drug_KG_features_normalized.csv"


df = pd.read_csv(input_file)


emb_data = df['emb'].apply(lambda x: np.array(list(map(float, x.split(',')))))


emb_matrix = np.stack(emb_data)


scaler = StandardScaler()
emb_normalized = scaler.fit_transform(emb_matrix)


df['emb'] = [','.join(map(str, row)) for row in emb_normalized]


df.to_csv(output_file, index=False)

print(f"标准化后的数据已保存到 {output_file}")





df = pd.read_csv('final_node_features(1).csv', header=None)


df.columns = ['drug'] + [f'feature_{i}' for i in range(1, 101)]


df['emb'] = df.iloc[:, 1:].apply(lambda row: ', '.join(map(str, row)), axis=1)


output_df = df[['drug', 'emb']]
output_df.to_csv('drug_KG_features.csv', index=False, header=['drug', 'emb'])