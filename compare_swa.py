import numpy as np
import pandas as pd
import os
os.chdir('data_long')

r = []
for a, b in [
    ('run-version_33-tag-val_acc.csv', 'run-version_224-tag-val_acc.csv'),
    ('run-version_40-tag-val_acc.csv', 'run-version_225-tag-val_acc.csv'),
    ('run-version_50-tag-val_acc.csv', 'run-version_227-tag-val_acc.csv'),
    ('run-version_57-tag-val_acc.csv', 'run-version_228-tag-val_acc.csv'),
    ('run-version_103-tag-val_acc.csv', 'run-version_229-tag-val_acc.csv'),
    ('run-version_104-tag-val_acc.csv', 'run-version_230-tag-val_acc.csv'),
    ('run-version_105-tag-val_acc.csv', 'run-version_231-tag-val_acc.csv'),
    ('run-version_106-tag-val_acc.csv', 'run-version_232-tag-val_acc.csv'),
    ('run-version_107-tag-val_acc.csv', 'run-version_233-tag-val_acc.csv'),
    ('run-version_108-tag-val_acc.csv', 'run-version_234-tag-val_acc.csv'),
    ('run-version_109-tag-val_acc.csv', 'run-version_235-tag-val_acc.csv')
]:
    take = int(len(pd.read_csv(b)) * 0.04)
    print(a, b, take)
    r.append(sum(pd.read_csv(b).values[:, 2][-take:]) / sum(pd.read_csv(a).values[:, 2][-take:]))

print(np.mean(r) * 100 - 100)
print(np.std(r) * 100)
