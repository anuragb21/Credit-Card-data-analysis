

```python
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
```


<style>.container { width:95% !important; }</style>



```python
df = pd.read_csv('CreditCard.csv')
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card</th>
      <th>reports</th>
      <th>age</th>
      <th>income</th>
      <th>share</th>
      <th>expenditure</th>
      <th>owner</th>
      <th>selfemp</th>
      <th>dependents</th>
      <th>months</th>
      <th>majorcards</th>
      <th>active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>no</td>
      <td>0</td>
      <td>25.83333</td>
      <td>1.5900</td>
      <td>0.000755</td>
      <td>0.0</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>87</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>no</td>
      <td>2</td>
      <td>25.91667</td>
      <td>2.0700</td>
      <td>0.000580</td>
      <td>0.0</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>no</td>
      <td>3</td>
      <td>40.50000</td>
      <td>4.0128</td>
      <td>0.000299</td>
      <td>0.0</td>
      <td>no</td>
      <td>no</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>no</td>
      <td>4</td>
      <td>57.75000</td>
      <td>2.0000</td>
      <td>0.000600</td>
      <td>0.0</td>
      <td>yes</td>
      <td>yes</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>no</td>
      <td>1</td>
      <td>63.41667</td>
      <td>2.1375</td>
      <td>0.000561</td>
      <td>0.0</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>240</td>
      <td>1</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['card', 'reports', 'age', 'income', 'share', 'expenditure', 'owner',
           'selfemp', 'dependents', 'months', 'majorcards', 'active'],
          dtype='object')




```python
df.dtypes
```




    card            object
    reports          int64
    age            float64
    income         float64
    share          float64
    expenditure    float64
    owner           object
    selfemp         object
    dependents       int64
    months           int64
    majorcards       int64
    active           int64
    dtype: object




```python
df['age'].describe()
```




    count    1082.000000
    mean       33.222427
    std        10.267899
    min         0.166667
    25%        25.333330
    50%        31.166670
    75%        39.666670
    max        83.500000
    Name: age, dtype: float64




```python
df['income'].describe()
```




    count    1082.000000
    mean        3.427969
    std         1.722428
    min         0.210000
    25%         2.300000
    50%         3.000000
    75%         4.000000
    max        13.500000
    Name: income, dtype: float64




```python
df['expenditure'].describe()
```




    count    1082.000000
    mean      225.591753
    std       284.950643
    min         0.000000
    25%        54.727295
    50%       141.017500
    75%       304.562300
    max      3099.505000
    Name: expenditure, dtype: float64




```python
df['share'].describe()
```




    count    1082.000000
    mean        0.083685
    std         0.098379
    min         0.000109
    25%         0.021228
    50%         0.054874
    75%         0.110494
    max         0.906320
    Name: share, dtype: float64




```python
df['card'].value_counts()
```




    yes    1023
    no       59
    Name: card, dtype: int64




```python
df['owner'].value_counts()
```




    no     569
    yes    513
    Name: owner, dtype: int64




```python
df['selfemp'].value_counts()
```




    no     1010
    yes      72
    Name: selfemp, dtype: int64




```python
df['card'] = pd.Series(np.where(df['card'] == 'yes', 1, 0),df.index)
```


```python
df['owner'] = pd.Series(np.where(df['owner'] == 'yes', 1, 0),df.index)
```


```python
df['selfemp'] = pd.Series(np.where(df['selfemp'] == 'yes', 1, 0),df.index)
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card</th>
      <th>reports</th>
      <th>age</th>
      <th>income</th>
      <th>share</th>
      <th>expenditure</th>
      <th>owner</th>
      <th>selfemp</th>
      <th>dependents</th>
      <th>months</th>
      <th>majorcards</th>
      <th>active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>25.83333</td>
      <td>1.5900</td>
      <td>0.000755</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>87</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>25.91667</td>
      <td>2.0700</td>
      <td>0.000580</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>40.50000</td>
      <td>4.0128</td>
      <td>0.000299</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4</td>
      <td>57.75000</td>
      <td>2.0000</td>
      <td>0.000600</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>63.41667</td>
      <td>2.1375</td>
      <td>0.000561</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>240</td>
      <td>1</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    card           0
    reports        0
    age            0
    income         0
    share          0
    expenditure    0
    owner          0
    selfemp        0
    dependents     0
    months         0
    majorcards     0
    active         0
    dtype: int64




```python
df.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card</th>
      <th>reports</th>
      <th>age</th>
      <th>income</th>
      <th>share</th>
      <th>expenditure</th>
      <th>owner</th>
      <th>selfemp</th>
      <th>dependents</th>
      <th>months</th>
      <th>majorcards</th>
      <th>active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1077</th>
      <td>1</td>
      <td>0</td>
      <td>30.58333</td>
      <td>2.512</td>
      <td>0.002627</td>
      <td>4.583333</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>36</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1078</th>
      <td>1</td>
      <td>0</td>
      <td>33.58333</td>
      <td>4.566</td>
      <td>0.002146</td>
      <td>7.333333</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>94</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1079</th>
      <td>1</td>
      <td>0</td>
      <td>40.58333</td>
      <td>4.600</td>
      <td>0.026513</td>
      <td>101.298300</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>1</td>
      <td>0</td>
      <td>32.83333</td>
      <td>3.700</td>
      <td>0.008999</td>
      <td>26.996670</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>60</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1081</th>
      <td>1</td>
      <td>0</td>
      <td>48.25000</td>
      <td>3.700</td>
      <td>0.111619</td>
      <td>344.157500</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
inputs = ['reports', 'age', 'income', 'share', 'expenditure', 'owner','selfemp', 'dependents', 'months', 'majorcards', 'active']
```


```python
X = df[inputs]
```


```python
y = df['card']
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=1)
```


```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)




```python
y_pred = logreg.predict(X_test)
```


```python
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
```

    0.986175115207
    


```python
print(metrics.confusion_matrix(y_test, y_pred))
```

    [[ 12   0]
     [  3 202]]
    


```python
print(metrics.accuracy_score(y_test, y_pred))
```

    0.986175115207
    


```python
print(metrics.recall_score(y_test, y_pred))
```

    0.985365853659
    


```python
confusion = metrics.confusion_matrix(y_test, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
```


```python
print(TN / float(TN + FP))
```

    1.0
    


```python
print(FP / float(TN + FP))
```

    0.0
    


```python
print(metrics.precision_score(y_test, y_pred))
```

    1.0
    


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier(n_neighbors=5)
```


```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-35-114a527a2c12> in <module>()
    ----> 1 scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
          2 print(scores.mean())
    

    NameError: name 'cross_val_score' is not defined

