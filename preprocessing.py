import pandas as pd
import json
import os
import glob

folder_path = 'metadata5'
json_files = glob.glob(os.path.join(folder_path, '*.json'))

data_list = []

for file in json_files:
    with open(file, 'r') as f:
        content = f.read()
        try:
            # Parse the JSON content
            json_data = json.loads(content)
            data_list.append(json_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing {file}: {e}")

df = pd.DataFrame(data_list)
print(df)

for col in df.columns:
    print(col)

df.to_csv('metadata-5.csv', sep=',', encoding='utf-8', index=False, header=True)