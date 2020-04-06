#%%
import os
import pandas as pd
import glob 

# %%
folder = '/Users/amsurve/PROJECTS/gg2/data/bbc/'

# %%
path_li = glob.glob(f'{folder}/**/*.txt' , recursive=True)

# %%
labels = {'tech':0,'business':1,'sport':2,'entertainment':3,'politics':4}
bbc_dict = []
for f in path_li:
    with open(f, 'rb') as myfile:
        data = myfile.read()
        data = "".join(map(chr, data))
        # print(data)
        # print('*'*50)
        bbc_dict.append({
            'category':f.split('/')[-2],
            'title':data.split('\n')[0],
            'text':data,
            'label':labels[f.split('/')[-2]]})
# %%
df = pd.DataFrame(bbc_dict)

# %%
if __name__ == '__main__':
    df.to_csv('/Users/amsurve/PROJECTS/gg2/data/bbc_df.csv')
