import json
import pickle
import numpy as np

# global variables
__location = None
__data_columns = None
__model = None


def load_saved_artifacts():
    print('loading artifact...')
    global __data_columns
    global __location
    global __model

    with open('./artifact/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[3:]

    with open('./artifact/bengaluru_house_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)


def get_location_names():
    return __location


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1  # đánh dấu 1 cho location mình chọn

    return round(__model.predict([x])[0], 2)


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Indira Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
