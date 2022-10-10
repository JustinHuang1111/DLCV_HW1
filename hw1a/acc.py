import pandas as pd

df = pd.read_csv('./pred/pred.csv')

correct = 0

for i, row in df.iterrows():
    if int(row['filename'].split('_')[0]) == row['label']:
        correct += 1
print(correct / len(df))
