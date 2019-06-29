# Creates the train_test_validation csv files
#Result of the script will be inside the data folder

import pandas as pd
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split,  get_stances_for_folds

#Training Dataset
train_data = DataSet()
folds,hold_out = kfold_split(train_data, training = 0.8, n_folds=1)
fold_stances, hold_out_stances = get_stances_for_folds(train_data,folds,hold_out)

#Test Dataset
test_data = DataSet("competition_test")
test_d = list()    
for stance in test_data.stances:
    test_d.append([stance['Headline'],stance['Body ID']])
    
print("Training Size: ",len(fold_stances[0]))
print("Validation Size: ",len(hold_out_stances))
print("Test Size: ",len(test_d))

train_df = pd.DataFrame(fold_stances[0])
validation_df = pd.DataFrame(hold_out_stances)
test_df = pd.DataFrame(test_d,columns = ["Headline","Body ID"])

print("Training Shape: ",train_df.shape)
print("Validation Shape: ",validation_df.shape)
print("Test Shape: ",test_df.shape)

train_df.to_csv("data/train.csv", encoding='utf-8', index=False)
validation_df.to_csv("data/validation.csv", encoding='utf-8', index=False)
test_df.to_csv("data/test.csv", encoding='utf-8', index=False)

train_bodyid = pd.read_csv("fnc-1/train_bodies.csv")
test_bodyid = pd.read_csv("fnc-1/competition_test_bodies.csv")
body_id = pd.concat([train_bodyid,test_bodyid])

body_id.sort_values(by=['Body ID'],inplace=True)
body_id.to_csv("data/body_id", encoding='utf-8', index=False)
body_id.shape
