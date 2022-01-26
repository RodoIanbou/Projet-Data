import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from catboost import CatBoostRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score




data_path='C:/Users/Rodolphe/Documents/COURS/2A/AADA/Projet_apprentissage_dynamique/martdata.zip'

# loading the dataset to a Pandas DataFrame

zf = zipfile.ZipFile(data_path)
mart_data_train= pd.read_csv(zf.open("Train.csv"))
mart_data_test= pd.read_csv(zf.open("Test.csv"))
                       
#mart_data_train.head()            # display the first 5 rows of the dataset           
#mart_data_test.head()

#mart_data_train.tail()             #display the last 5 rows of the dataset
#mart_data_test.tail()

#info_test=mart_data_test.info()     #display information about the dataset
#info_train=mart_data_train.info()


missing_values_test=mart_data_test.isnull().sum()          #Dans les colonnes item_weight et outlet_sie, il ya des valeurs manquantes               
missing_values_train=mart_data_train.isnull().sum()


#nettoyage des données

data_train=mart_data_train.dropna()           #On supprime les lignes avec des valeurs manquantes
data_test=mart_data_test.dropna()



#Nettoyage mart_data_train

mean_Item_Weight_train = data_train.Item_Weight.describe()[1]             
mart_data_train.Item_Weight.fillna(mean_Item_Weight_train, inplace = True)

mean_Outlet_Size_train = 0
for i in range(len(mart_data_train)):
    if(mart_data_train.Outlet_Size[i] == "Small"):
        mean_Outlet_Size_train+=1
        mart_data_train.Outlet_Size[i] = 1
    if(mart_data_train.Outlet_Size[i] == "Medium"):
        mean_Outlet_Size_train+=2
        mart_data_train.Outlet_Size[i] = 2
    if(mart_data_train.Outlet_Size[i] == "High"):
        mean_Outlet_Size_train+=3
        mart_data_train.Outlet_Size[i] = 3
mean_Outlet_Size_train = mean_Outlet_Size_train/6113

        
mart_data_train.Outlet_Size.fillna(mean_Outlet_Size_train, inplace = True)



#Nettoyage mart_data_test

mean_Item_Weight_test = data_test.Item_Weight.describe()[1]
mart_data_test.Item_Weight.fillna(mean_Item_Weight_test, inplace = True)

mean_Outlet_Size_test = 0
for i in range(len(mart_data_test)):
    if(mart_data_test.Outlet_Size[i] == "Small"):
        mean_Outlet_Size_test+=1
        mart_data_test.Outlet_Size[i] = 1
    if(mart_data_test.Outlet_Size[i] == "Medium"):
        mean_Outlet_Size_test+=2
        mart_data_test.Outlet_Size[i] = 2
    if(mart_data_test.Outlet_Size[i] == "High"):
        mean_Outlet_Size_test+=3
        mart_data_test.Outlet_Size[i] = 3
mean_Outlet_Size_test = mean_Outlet_Size_test/4075

        
mart_data_test.Outlet_Size.fillna(mean_Outlet_Size_test, inplace = True)



mart_data_train.assign('')


X = mart_data_train.drop(columns = 'Item_Outlet_Sales', axis = 1)
Y = mart_data_train['Item_Outlet_Sales']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)





#description statistique des données
Description_train=mart_data_train.Item_Weight.describe()
Description_test=mart_data_test.Item_Weight.describe()

categorical_features_indices = np.where(X.dtypes != np.float)[0]

model=CatBoostRegressor(iterations=100, depth=4, learning_rate=0.05, loss_function='RMSE')
model.fit(x_train, y_train,cat_features=categorical_features_indices,eval_set=(x_test, y_test),plot=True)


prediction_train = model.predict(x_test)
R2 = r2_score(y_test, prediction_train)


prediction_test = pd.DataFrame()
prediction_test['Item_Identifier'] = mart_data_test['Item_Identifier']
prediction_test['Outlet_Identifier'] = mart_data_test['Outlet_Identifier']
prediction_test['Item_Outlet_Sales'] = model.predict(mart_data_test)
prediction_test.to_csv("Submission.csv")