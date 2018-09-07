import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
import os.path
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

myPath = os.path.abspath(os.path.dirname(__file__))
data = pd.read_csv(os.path.join(myPath, "banking.csv"), header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

data.head()

data['education'].unique()

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

data['education'].unique()

data['y'].value_counts()

sns.countplot(x='y',data=data, palette='hls')
plt.show()
plt.savefig('count_plot')

data.groupby('y').mean()

data.groupby('job').mean()

data.groupby('marital').mean()

data.groupby('education').mean()


pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')

