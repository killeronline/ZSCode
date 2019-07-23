import sys # analysis:ignore
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier           # analysis:ignore
from sklearn.linear_model import LogisticRegression           # analysis:ignore


# Turbo
# date_of_game > day of week, day of month, month of year
# dropping half of train dataset because of nan, use correlation df.corr()

df = pd.read_csv('data.csv',index_col=0)
df = df.drop(['team_id','team_name'],axis=1)
df = df.dropna(subset=['shot_id_number'])
df = df.reset_index(drop=True)

label = 'is_goal'

categories = ['match_event_id',              
              'game_season',
              'area_of_shot',
              'shot_basics',
              'range_of_shot',
              'date_of_game',
              'home/away',              
              'lat/lng',
              'type_of_shot',
              'type_of_combined_shot',
              'match_id']

m = df.shape[0]
for c in categories:
    d = {}    
    s = set(df[c].astype(str).unique())
    for (v,k) in enumerate(s):
        d[k] = v    
    df[c] = df[c].map(d)
    
'''
effective_columns = ['distance_of_shot',
                     'range_of_shot',
                     'location_y']

                   

all_columns = df.columns
waste_columns = []
for c in all_columns :
    if c != label and c not in effective_columns :
        waste_columns.append(c)

'''
waste_columns = categories
waste_columns.remove('area_of_shot')
waste_columns.remove('shot_basics')
waste_columns.remove('range_of_shot')
waste_columns.remove('type_of_shot')
waste_columns.remove('type_of_combined_shot')

waste_columns.append('shot_id_number') # for results we can store before dropping
#waste_columns.append('knockout_match')
#waste_columns.append('knockout_match.1')
waste_columns.append('remaining_sec')
waste_columns.append('remaining_sec.1')
waste_columns.append('match_id')
waste_columns.append('match_event_id')
        
df = df.drop(waste_columns,axis=1) 

#df = df.drop(['match_id','match_event_id'],axis=1)

m = df.shape[0]

before = m - df.count()

import correlation_impute

correlation_impute.impute(df,label)

after = m - df.count()

#df = df.dropna()    


'''       
df = df.dropna()    
dcr = df.corr()
dcr = abs(dcr)
dcr = dcr[label].sort_values()
'''
           
units = df.copy()[pd.isnull(df[label])] # tests for final submission
units = units.drop([label],axis=1)
#units = units.dropna()
#units = units.fillna(value=-1)
units = units.fillna(units.mean())
units = units.reset_index(drop=True)

xdatas = df.copy()[pd.notnull(df[label])] # actual labelled data

sys.exit()

#xdatas = xdatas.dropna()
#xdatas = xdatas.fillna(value=-1)
xdatas = xdatas.fillna(xdatas.mean())
xdatas = xdatas.reset_index(drop=True)
ydatas = pd.DataFrame(xdatas.copy()[label])
xdatas = xdatas.drop([label],axis=1)


m = xdatas.shape[0] # samples
split = 85 # percentage split between train set and test set
splitindex = int(m*(split/100))

xtrain = xdatas[:splitindex]
ytrain = ydatas[:splitindex]

xtests = xdatas[splitindex:]
ytests = ydatas[splitindex:]

best_acc = 0 
n = xtrain.shape[1]

'''
clf = LogisticRegression()
clf.fit(xtrain,ytrain)
ypreds = clf.predict(xtests)
acc = metrics.accuracy_score(ytests, ypreds)
print('n:',n,'accuracy:',acc)

ypred2 = clf.predict(xtrain)
acc = metrics.accuracy_score(ytrain, ypred2)
print('n:',n,'accuracy:',acc)   

'''
#while n > 0 :
#if True :    

clf = RandomForestClassifier(n_estimators=1000,
                                     class_weight='balanced',
                                     max_features='auto',
                                     min_samples_leaf=0.01,
                                     max_depth=20,                                     
                                     criterion='gini',
                                     random_state=1,
                                     verbose=False,
                                     n_jobs=-1)
#clf.fit(xtrain,ytrain[label])
clf.fit(xtrain,ytrain)

ypreds_train = clf.predict(xtrain)
acc_train = metrics.accuracy_score(ytrain, ypreds_train)

ypreds = clf.predict(xtests)
acc = metrics.accuracy_score(ytests, ypreds)    

print('Train Accuracy:',acc_train)
print('Tests Accuracy:',acc)
    
    
estimator = clf.estimators_[5]
from sklearn.tree import export_graphviz

export_graphviz(estimator, out_file='tree.dot',
            feature_names = xtrain.columns,
            class_names = clf.classes_.astype(int).astype(str),
            rounded = True, 
            proportion = False, 
            precision = 2, 
            filled = True)
'''
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)

from IPython.display import Image
Image(graph.create_png())

'''
    
# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
    
'''
if False :
    imp = clf.feature_importances_
    lenImp = len(imp)
    least = 1
    least_fti = 0
    for fti in range(lenImp):
        if imp[fti] <= least :
            least = imp[fti]
            least_fti = fti
    pruning_column = xtrain.columns[least_fti]
    xtrain = xtrain.drop(pruning_column,axis=1)
    xtests = xtests.drop(pruning_column,axis=1)    
    n = xtrain.shape[1]        
    break
    '''
    

#xpredf = df[np.isnan(df['is_goal'])==True] # Rows that need prediction
#xdataf = df[np.isnan(df['is_goal'])==False] # Labelled data

'''
df['game_season'] = df['game_season'].astype(str)
#df['game_season'] = list(map(str,df['game_season']))
nankey = -999

cid = 26

cname = df.columns[cid]
kf = df[cname]
d = {}
d[nankey] = 0
for i in kf :
    if type(i) != str and np.isnan(i):        
        d[nankey] += 1
    elif str(i) in d :
        d[str(i)] += 1    
    else :
        d[str(i)] = 1
    
'''