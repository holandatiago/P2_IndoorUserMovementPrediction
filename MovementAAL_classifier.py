import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

N = 314; inCol = ['RSS_anchor1','RSS_anchor2','RSS_anchor3','RSS_anchor4']

input = np.array([np.mean(np.loadtxt('MovementAAL\dataset\MovementAAL_RSS_'+str(x+1)+'.csv',delimiter=','),axis=0) for x in range(N)]).tolist()
output = np.loadtxt('MovementAAL\dataset\MovementAAL_target.csv',int,delimiter=',',usecols=(1,)).tolist()

data = pd.DataFrame([input[x]+[output[x]] for x in range(N)],columns=inCol+['target'])
train = data[::2]; test = data[1::2]

clf = DecisionTreeClassifier(random_state=0)
clf.fit(train[inCol],train['target'])

score = roc_auc_score(clf.predict(test[inCol]),test['target'])
print('Test data AUC score:', score)

pred = np.stack((np.arange(1,N+1),clf.predict(data[inCol])),axis=-1)
np.savetxt('MovementAAL_prediction.csv',pred,'%d',',',header='sequence_ID, predicted_label')