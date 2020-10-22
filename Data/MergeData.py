import pandas as pd
import numpy as np
import os
import operator


def file_filter(file_list, contain_text):
    res = []
    for file in file_list:
        if contain_text in file:
            res.append(file)
    return res


train_files = file_filter(os.listdir('./train/'), 'indicator')
test_files = file_filter(os.listdir('./test/'), 'indicator')
assert operator.eq(train_files, test_files)

for file in train_files:
    train_file = pd.read_csv(f'./train/{file}', index_col=0)
    test_file = pd.read_csv(f'./test/{file}', index_col=0)
    merged_file = pd.concat([train_file, test_file], axis=0, sort=False)
    merged_file.to_csv(f'./raw/{file}')
