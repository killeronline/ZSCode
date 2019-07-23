import numpy as np
import pandas as pd
import random as rd
from sklearn import metrics
from sklearn.linear_model import LinearRegression           # analysis:ignore


'''
a = list(range(50))

for i in range(len(a)):
    t = a[i] * a[i]
    p = a[i] * 9
    r = rd.randint(0,p)
    j = a[i]
    if t % 3 == 0 :
        t = np.nan
    if p % 4 == 0 :
        p = np.nan
    if r % 5 == 0 :
        r = np.nan
    if j % 6 == 0 :
        j = np.nan
    a[i] = [j,t,p,r,t+p]
        
df = pd.DataFrame(a)    
df.columns = ['number','square','product9','random','mix']    

target = 'mix'
'''

def impute(df,target):
    corr_limit = df.shape[1]
    corr_count = 0
    corr_pairs = {}
    while corr_count < corr_limit :
        corr_count += 1
        
        # correlate features
        dcr = df.corr()
        
        # end goal is to fill the mix column by imputing data into square and product cols
        
        # find the best pair from correlation matrix and try linear regression on them
        feature_columns = df.columns
        dcr = np.array(dcr).astype(float)
        n = len(feature_columns)
        
        best_correlation = 0
        best_i = 0
        best_j = 0
        for i in range(n):                
            for j in range(i+1,n):
                c = dcr[i][j]
                corr_key = 'Ft{}_Ft{}'.format(i,j)
                rule1 = c > best_correlation and corr_key not in corr_pairs
                rule2 = feature_columns[i] != target
                rule3 = feature_columns[j] != target
                rule4 = df[pd.isnull(df[feature_columns[j]])].shape[0]>0                        
                rule5 = c > 0.6 # minimum correlation needed
                if rule1 and rule2 and rule3 and rule4 and rule5 :
                #if rule1 :
                    print(i,j,rule1,rule2,rule3,rule4,c)
                    best_correlation = c
                    best_i = i
                    best_j = j                   
            
        if best_correlation == 0 :
            break    
        else :               
            corr_key = 'Ft{}_Ft{}'.format(best_i,best_j)
            corr_pairs[corr_key] = 1
            double_impute = [[best_i,best_j],[best_j,best_i]]
            for (best_i,best_j) in double_impute :
                p = feature_columns[best_i]
                q = feature_columns[best_j]            
                print('Imputing {} from {}'.format(q,p),best_correlation,best_i,best_j)
                pf = df.copy()[[p,q]]    
                # finding q from p, also should do vice versa    
                pf = pf.dropna(subset=[p])            
                # include higher degree polynomials (till cubic and quadratic)
                pf[p+'^2'] = pf[p] * pf[p]
                pf[p+'^3'] = pf[p] * pf[p] * pf[p]
                pf[p+'^4'] = pf[p] * pf[p] * pf[p] * pf[p]
                
                units = pf[pd.isnull(pf[q])]
                units = units.drop([q],axis=1)            
                #units = units.reset_index(drop=True)
                
                xdatas = pf[pd.notnull(df[q])] # actual labelled data
                xdatas = xdatas.reset_index(drop=True)
                ydatas = pd.DataFrame(xdatas.copy()[q])
                xdatas = xdatas.drop([q],axis=1)
            
                m = xdatas.shape[0] # samples
                split = 85 # percentage split between train set and test set
                splitindex = int(m*(split/100))
            
                xtrain = xdatas[:splitindex]
                ytrain = ydatas[:splitindex]
                
                xtests = xdatas[splitindex:]
                ytests = ydatas[splitindex:]
    
                if len(units) == 0 or len(xtrain) == 0 or len(xtests) == 0 :
                    continue # Not Enough Data to Fit
                
                reg = LinearRegression()
                reg.fit(xtrain,ytrain)
                
                
                ypreds_train = reg.predict(xtrain)
                mae_train = metrics.mean_absolute_error(ytrain, ypreds_train)            
                        
                ypreds = reg.predict(xtests)
                mae = metrics.mean_absolute_error(ytests, ypreds)            
                        
                print('Train Mae:',mae_train)
                print('Tests Mae:',mae)
                
                
                ypreds_units = reg.predict(units)
                yunits = pd.DataFrame(ypreds_units)
                yunits.index = units.index
                yunits.columns = ytrain.columns
                
                # Fill Up NaN's    
                for i in units.index:
                    df[q][i] = yunits[q][i]
    
#impute(df,target)
    

























