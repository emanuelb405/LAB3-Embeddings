#This file does the preprocessing and train test split of the pictures downloaded
#from kaggle https://www.kaggle.com/c/dogs-vs-cats

from os import listdir
import pandas as pd
import re
files = pd.Series(listdir('train'))
df = pd.DataFrame()
dog = files[files.apply(lambda x:bool(re.match('dog',x)))]
#dog = dog[dog != "NaN"]
dog = dog.reset_index()

cat = files[files.apply(lambda x:bool(re.match('cat',x)))]
#cat = cat[cat != 'NaN']
cat = cat.reset_index()
df['dog']=dog[0]
df['cat']=cat[0]

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.33, random_state=42,shuffle =True)

len(X_test['dog'])

##create train folder
from shutil import copyfile
import os
for file in X_train['dog']:
  print(file)
  dirname ='train_split/dog/'
  filename = str(dirname+file)
  if not os.path.exists(os.path.dirname(dirname)):
    os.makedirs(os.path.dirname(dirname))
  copyfile(str('train/'+file),str('train_split/dog/'+file))
  
from shutil import copyfile
import os
for file in X_train['cat']:
  print(file)
  dirname ='train_split/cat/'
  filename = str(dirname+file)
  if not os.path.exists(os.path.dirname(dirname)):
    os.makedirs(os.path.dirname(dirname))
  copyfile(str('train/'+file),str('train_split/cat/'+file))
  
##create test folder
from shutil import copyfile
import os
for file in X_test['dog']:
  print(file)
  dirname ='test_split/dog/'
  filename = str(dirname+file)
  if not os.path.exists(os.path.dirname(dirname)):
    os.makedirs(os.path.dirname(dirname))
  copyfile(str('train/'+file),str('test_split/dog/'+file))
  
from shutil import copyfile
import os
for file in X_test['cat']:
  print(file)
  dirname ='test_split/cat/'
  filename = str(dirname+file)
  if not os.path.exists(os.path.dirname(dirname)):
    os.makedirs(os.path.dirname(dirname))
  copyfile(str('train/'+file),str('test_split/cat/'+file))