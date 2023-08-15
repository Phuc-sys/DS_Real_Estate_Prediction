import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import json


''' Data Cleaning '''
df = pd.read_csv('bengaluru_house_prices.csv')
print(df.shape)
print(df.groupby('area_type')['area_type'].agg('count'))
df1 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(df1.head())

print(df1.isnull().sum())
df2 = df1.dropna()
print(df2['size'].unique())
df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
print(df2.head())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
print(df2[~df2['total_sqft'].apply(is_float)].head())  # check invalid value

def convert_sqft_num(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None
df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_num)
df3['price_per_sqft'] = df3['price'] * 100000 / df3['total_sqft']
print(df3.head())

''' Feature Engineer '''
print(len(df3.location.unique()))  # 1304 unique feature -> quá lớn nếu dùng 1 hot encoding sẽ tạo ra thêm 1304 columns
df3.location = df3.location.apply(lambda x: x.strip())
location_stats = df3.groupby('location')['location'].agg('count').sort_values(ascending=False) # xác định tần suất xuất hiện của mỗi location
print(location_stats)
location_stats_less_than_10 = location_stats[location_stats < 10]
print(len(location_stats_less_than_10))
df3.location = df3.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x) # chỉ lấy location xuất hiện >= 10
print(len(df3.location.unique()))

''' Outlier Remove '''
print(df3.shape)
df4 = df3[~(df3.total_sqft / df3.bhk < 300)] # 1 bhk at least equal 300 sqft, else remove
print(df4.shape)
print(df4.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        df_reduced = subdf[(subdf.price_per_sqft > (mean-std)) & (subdf.price_per_sqft < (mean+std))]
        df_out = pd.concat([df_out, df_reduced], ignore_index=True)
    return df_out
df5 = remove_pps_outliers(df4)
print(df5.shape)
df5 = df5[df5.bath < df5.bhk + 2]
print(df5.shape)

''' Model Building '''
df6 = df5.drop(['size', 'price_per_sqft'], axis='columns')
dummies = pd.get_dummies(df6.location)
dummies.drop('other', axis='columns', inplace=True)
df7 = pd.concat([df6, dummies], axis='columns')
df7.drop('location', axis='columns', inplace=True)
print(df7.head(3))

x = df7.drop('price', axis='columns')
y = df7.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(cross_val_score(LinearRegression(), x, y, cv=cv))

def find_best_model(x, y):
    algo = {
        'linear_reg': {
            'model': LinearRegression(),
            'params': {
                #'normalize': [True, False]
                #StandardScaler()
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
         },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algo.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
best_model = find_best_model(x, y)
print(best_model)

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(x.columns==location)[0][0] # return index column của location đó
    X = np.zeros(len(x.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1 # đánh dấu 1 cho location mình chọn
    return model.predict([X])[0]

print(predict_price('Indira Nagar',1000, 2, 2))
# save model
with open('bengaluru_house_prices_model.pickle', 'wb') as f:
    pickle.dump(model, f)
# save columns info
columns = {
    'data_columns': [col.lower() for col in x.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))