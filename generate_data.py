import pandas as pd
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split,  get_stances_for_folds

get_dataFrame(data):
    d = list()
    for stance in data.stances:
        s = stance['Stance']
        # Relatedness
        if s == 'unrelated':
          a = s
          b=None
          c=None
        else:
          a = 'related'
          ## Decision or Discuss
          if s == 'discuss':
            b = s
            c=None
          else:
            b = 'ANA'
            c = s

        d.append([stance['Headline'],data.articles[stance['Body ID']],s,a,b,c])
        df = pd.DataFrame(d,columns = ['Headline','Body','Stance','Relatedness','Discussion','AgreeNotagree'])
    return df

#Training Dataset
train_data = DataSet()
train_df = get_dataFrame(train_data)
train_df.to_csv("data/train.csv", encoding='utf-8', index=False)    

#Training Dataset
test_data = DataSet("competition_test")
test_df = get_dataFrame(test_data)
test_df.to_csv("data/test.csv", encoding='utf-8', index=False)    
