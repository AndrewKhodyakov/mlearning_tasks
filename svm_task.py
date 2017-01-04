import pandas as pd
import matplotlib
import seaborn
from sklearn import svm

matplotlib.use('Agg')

if __name__ == "__main__":
    data = pd.read_csv(\
        './datasets/_f6284c13db83a3074c2b987f714f24f5_svm-data.csv',
        names=['target', 'A', 'B'])

    S = svm.SVC(C=100000, random_state=241, kernel='linear')
    svm_model = S.fit(data[['A', 'B']], data['target'])
    f = open('./svm_result.txt', 'wt')
    res = ''
    tmp = svm_model.support_
    tmp.sort()
    for i in tmp:
        res = res + str(i+1) + ','
    f.write(res)
    f.close()

    plt = seaborn.regplot(data['A'], data['B'])
    plt.figure.savefig('./svm.png')
