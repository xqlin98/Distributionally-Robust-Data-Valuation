import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.neighbors import KNeighborsRegressor
import torchvision
import torchvision.transforms as transforms
import copy

def load_rideshare(numdp, seed, **kwargs):
    # load data
    df = pd.read_csv('./dataset/rideshare_kaggle.csv')

    # Remove empty cell rows
    df = df[df['price'].isnull() == False]

    # Feature selection
    df = df[['price', 'distance', 'surge_multiplier', 'day', 'month', 'windBearing', 'cloudCover', 'name']]
    df = pd.get_dummies(df,columns=['name'], drop_first=True)
    df = df.drop(['name_Lux', 'name_Lux Black', 'name_Lux Black XL', 'name_Lyft', 'name_Lyft XL'], axis=1)

    # Shuffle and train test split
    data = df.copy()
    data = shuffle(data)
    X = data.drop(['price'], axis=1).values
    Y = data['price'].values

    # 70/30 Split should do
    trdata, tedata, trlabel, telabel = train_test_split(X,Y,test_size=15000, train_size=numdp,random_state=seed)

    scaler = StandardScaler()
    scaler.fit(X)
    trdata, tedata = scaler.transform(trdata), scaler.transform(tedata)
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)

    return trdata, tedata, trlabel, telabel

def load_mnist(numdp, seed, flatten=True, **kwargs):
    (train_x_raw, train_y_raw), (test_x_raw, test_y_raw) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    
    train_x_raw, test_x_raw = train_x_raw.reshape(train_x_raw.shape[0],-1), test_x_raw.reshape(test_x_raw.shape[0],-1)
    scaler = StandardScaler()
    scaler.fit(train_x_raw)
    train_x_raw, test_x_raw = scaler.transform(train_x_raw), scaler.transform(test_x_raw)
    
    if not flatten:
        train_x_raw, test_x_raw = train_x_raw.reshape(train_x_raw.shape[0], 1, 28 , 28), test_x_raw.reshape(test_x_raw.shape[0],1 ,28 , 28)
    train_y_raw, test_y_raw = np.eye(10)[train_y_raw], np.eye(10)[test_y_raw]

    # split test and train dataset
    trdata, _, trlabel, _  = train_test_split(train_x_raw, train_y_raw,test_size=50000, train_size=numdp,random_state=seed)
    tedata, telabel = test_x_raw, test_y_raw

    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    
    # print(np.isnan(np.sum(trdata)), np.isfinite(np.sum(trdata)),np.isnan(np.sum(tedata)), np.isfinite(np.sum(tedata)))
    # raise NotImplementedError

    return trdata, tedata, trlabel, telabel


def load_cifar10(numdp, seed, flatten=True, **kwargs):

    # Data
    print('==> Preparing data..')

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    means = np.array([0.4914, 0.4822, 0.4465]).reshape([1,1,1,3])
    stds = np.array([0.2023, 0.1994, 0.2010]).reshape([1,1,1,3])

    train_data = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=True)

    test_data = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=True)
    trdata, trlabel, tedata, telabel = train_data.data/255, train_data.targets, test_data.data/255, test_data.targets
    trdata, tedata = (trdata - means)/stds, (tedata - means)/stds
    trdata, tedata = trdata.transpose([0,3,1,2]), tedata.transpose([0,3,1,2])
    if flatten:
        trdata, tedata = trdata.reshape(trdata.shape[0],-1), tedata.reshape(tedata.shape[0],-1)
        
    trlabel, telabel = np.eye(10)[trlabel], np.eye(10)[telabel]

    # split test and train dataset
    trdata, _, trlabel, _  = train_test_split(trdata, trlabel,test_size=10000, train_size=numdp,random_state=seed)
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    
    # print(np.isnan(np.sum(trdata)), np.isfinite(np.sum(trdata)),np.isnan(np.sum(tedata)), np.isfinite(np.sum(tedata)))
    # raise NotImplementedError
    # raise KeyError

    return trdata, tedata, trlabel, telabel

def load_housing(numdp, seed, **kwargs):
    # function that imputes a dataframe 
    def impute_knn(df):
        ''' inputs: pandas df containing feature matrix '''
        ''' outputs: dataframe with NaN imputed '''
        # imputation with KNN unsupervised method

        # separate dataframe into numerical/categorical
        ldf = df.select_dtypes(include=[np.number])           # select numerical columns in df
        ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
        # define columns w/ and w/o missing data
        cols_nan = ldf.columns[ldf.isna().any()].tolist()         # columns w/ nan 
        cols_no_nan = ldf.columns.difference(cols_nan).values     # columns w/o nan 

        for col in cols_nan:                
            imp_test = ldf[ldf[col].isna()]   # indicies which have missing data will become our test set
            imp_train = ldf.dropna()          # all indicies which which have no missing data 
            model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
            knr = model.fit(imp_train[cols_no_nan], imp_train[col])
            ldf.loc[df[col].isna(), col] = knr.predict(imp_test[cols_no_nan])
        
        return pd.concat([ldf,ldf_putaside],axis=1)
    df = pd.read_csv('./dataset/housing.csv')
    del df['ocean_proximity']

    # Call function that imputes missing data
    df2 = impute_knn(df)
    # looks like we have a full feature matrix
    df2.info()

    # split test and train dataset
    trdata, tedata = train_test_split(df2,test_size=15000, train_size=numdp,random_state=seed)

    maxval2 = trdata['median_house_value'].max() # get the maximum value
    trdata_upd = trdata[trdata['median_house_value'] != maxval2] 
    tedata_upd = tedata[tedata['median_house_value'] != maxval2]
    # Make a feature that contains both longtitude & latitude
    trdata_upd['diag_coord'] = (trdata_upd['longitude'] + trdata_upd['latitude'])         # 'diagonal coordinate', works for this coord
    trdata_upd['bedperroom'] = trdata_upd['total_bedrooms']/trdata_upd['total_rooms']     # feature w/ bedrooms/room ratio
    # update test data as well
    tedata_upd['diag_coord'] = (tedata_upd['longitude'] + tedata_upd['latitude'])
    tedata_upd['bedperroom'] = tedata_upd['total_bedrooms']/tedata_upd['total_rooms']     # feature w/ bedrooms/room ratio

    alldata = df2
    del alldata['median_house_value']
    trlabel, telabel = trdata['median_house_value'], tedata['median_house_value']
    del trdata['median_house_value'], tedata['median_house_value']

    scaler = StandardScaler()
    scaler.fit(alldata)
    trdata, tedata = scaler.transform(trdata), scaler.transform(tedata)
    
    label_mean, label_std = np.mean(telabel), np.std(telabel)
    trlabel, telabel = (trlabel - label_mean)/label_std, (telabel - label_mean)/label_std
    
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)

    return trdata, tedata, trlabel, telabel


def load_used_car(numdp, seed, **kwargs):

    PATH = 'dataset/Used_car/'
    
    # Load data and shuffle
    audi_df = shuffle(pd.read_csv(PATH + 'audi.csv'))
    toyota_df = shuffle(pd.read_csv(PATH + 'toyota.csv'))
    ford_df = shuffle(pd.read_csv(PATH + 'ford.csv'))
    bmw_df = shuffle(pd.read_csv(PATH + 'bmw.csv'))
    vw_df = shuffle(pd.read_csv(PATH + 'vw.csv'))
    mercedez_df = shuffle(pd.read_csv(PATH + 'merc.csv'))
    vauxhall_df = shuffle(pd.read_csv(PATH + 'vauxhall.csv'))
    skoda_df = shuffle(pd.read_csv(PATH + 'skoda.csv'))

    # Identifier
    audi_df['model'] = 'audi'
    toyota_df['model'] = 'toyota'
    ford_df['model'] = 'ford'
    bmw_df['model'] = 'bmw'
    vw_df['model'] = 'vw'
    mercedez_df['model'] = 'mercedez'
    vauxhall_df['model'] = 'vauxhall'
    skoda_df['model'] = 'skoda'

    car_manufacturers = pd.concat([audi_df,
                                   toyota_df,
                                   ford_df,
                                   bmw_df,
                                   vw_df, 
                                   mercedez_df,
                                   vauxhall_df,
                                   skoda_df,
                                   ])
    
    # Remove invalid value rows
    car_manufacturers = car_manufacturers[car_manufacturers['year'] <= 2021]
    
    # Feature selection
    X = car_manufacturers[['year', 'mpg', 'mileage', 'tax', 'engineSize']].values
    y = car_manufacturers['price'].values.reshape(-1)

    # Train test split
    trdata, tedata, trlabel, telabel = train_test_split(X,y,test_size=15000, train_size=numdp,random_state=seed)

    scaler = StandardScaler()
    scaler.fit(X)
    trdata, tedata = scaler.transform(trdata), scaler.transform(tedata)
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    
    return trdata, tedata, trlabel, telabel

def load_uber_lyft(numdp, seed, **kwargs):
    """     
	    Method to load the Uber_lyft dataset.
        Args:
            n_participants (int): number of data subsets to generate
            s (int): number of data samples for each participant (equal)
            reduced (bool): whether to use a reduced csv file for faster loading
            path_prefix (str): prefix for the file path
        Returns:
            feature_datasets, labels, feature_datasets_test, test_labels: each a list containing the loaded dataset
    """
    
    df = pd.read_csv('dataset/Uber_lyft/rideshare_kaggle.csv')
    
    # Remove empty cell rows
    df = df[df['price'].isnull() == False]
    
    # Feature selection
    df = df[['price', 'distance', 'surge_multiplier', 'day', 'month', 'windBearing', 'cloudCover', 'name']]
    df = pd.get_dummies(df,columns=['name'], drop_first=True)
    df = df.drop(['name_Lux', 'name_Lux Black', 'name_Lux Black XL', 'name_Lyft', 'name_Lyft XL'], axis=1)

    # Shuffle and train test split
    data = df.copy()
    data = shuffle(data)
    X = data.drop(['price'], axis=1).values
    Y = data['price'].values

    trdata, tedata, trlabel, telabel = train_test_split(X,Y,test_size=30000, train_size=numdp,random_state=seed)

    scaler = StandardScaler()
    scaler.fit(X)
    trdata, tedata = scaler.transform(trdata), scaler.transform(tedata)

    # label_mean, label_std = np.mean(telabel), np.std(telabel)
    # trlabel, telabel = (trlabel - label_mean)/label_std, (telabel - label_mean)/label_std
    
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    return trdata, tedata, trlabel, telabel

def load_credit_card(numdp, seed, **kwargs):
    """     
	    Method to load the credit_card dataset.
        Args:
            n_participants (int): number of data subsets to generate
            s (int): number of data samples for each participant (equal)
            train_test_diff_distr (bool): whether to generate a test set that has a different distribution from the train set
            path_prefix (str): prefix for the file path
        Returns:
            feature_datasets, labels, feature_datasets_test, test_labels: each a list containing the loaded dataset
    """
    data = pd.read_csv('dataset/creditcard.csv')

    # Drop redundant features
    data = data.drop(['Class', 'Time'], axis = 1)
    
    data = shuffle(data)
    X = data.iloc[:, data.columns != 'Amount']
    y = data.iloc[:, data.columns == 'Amount'].values.reshape(-1)

    # Feature selection
    cols = ['V1', 'V2', 'V5', 'V7', 'V10', 'V20', 'V21', 'V23']
    X = X[cols]
    
    trdata, tedata, trlabel, telabel = train_test_split(X,y,test_size=30000, train_size=numdp,random_state=seed)

    scaler = StandardScaler()
    scaler.fit(X)
    trdata, tedata = scaler.transform(trdata), scaler.transform(tedata)

    label_mean, label_std = np.mean(telabel), np.std(telabel)
    trlabel, telabel = (trlabel - label_mean)/label_std, (telabel - label_mean)/label_std
    
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    return trdata, tedata, trlabel, telabel

def load_diabetes(numdp, seed, **kwargs):
    
    # Note that here we load the extracted features. Code for extracting the features can be found in the notebooks folder.
    loaded = np.load('dataset/diabetes/diabetic_data.npz', allow_pickle=True)
   
    # Shuffle
    X, y = loaded['X'], loaded['y'].astype(np.float64)

    trdata, tedata, trlabel, telabel = train_test_split(X,y,test_size=20000, train_size=numdp,random_state=seed)

    scaler = StandardScaler()
    scaler.fit(X)
    trdata, tedata = scaler.transform(trdata), scaler.transform(tedata)
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    return trdata, tedata, trlabel, telabel

def load_mnist_save(numdp, seed, flatten=True, **kwargs):
    (train_x_raw, train_y_raw), (test_x_raw, test_y_raw) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    num_train_samples = train_x_raw.shape[0]
    local_rng = np.random.default_rng(seed=seed)
    sampled_idx = local_rng.choice(num_train_samples, numdp, replace=False)
    trdata_raw, trlabel_raw = copy.deepcopy(train_x_raw[sampled_idx]), copy.deepcopy(train_y_raw[sampled_idx])
    trdata_raw, trlabel_raw = np.array(trdata_raw), np.array(trlabel_raw)

    train_x_raw, test_x_raw = train_x_raw.reshape(train_x_raw.shape[0],-1), test_x_raw.reshape(test_x_raw.shape[0],-1)
    scaler = StandardScaler()
    scaler.fit(train_x_raw)
    train_x_raw, test_x_raw = scaler.transform(train_x_raw), scaler.transform(test_x_raw)
    
    if not flatten:
        train_x_raw, test_x_raw = train_x_raw.reshape(train_x_raw.shape[0], 1, 28 , 28), test_x_raw.reshape(test_x_raw.shape[0],1 ,28 , 28)
    train_y_raw, test_y_raw = np.eye(10)[train_y_raw], np.eye(10)[test_y_raw]

    # split test and train dataset
    trdata, trlabel = train_x_raw[sampled_idx], train_y_raw[sampled_idx]
    tedata, telabel = test_x_raw, test_y_raw

    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    
    # print(np.isnan(np.sum(trdata)), np.isfinite(np.sum(trdata)),np.isnan(np.sum(tedata)), np.isfinite(np.sum(tedata)))
    # raise NotImplementedError

    return trdata, tedata, trlabel, telabel, trdata_raw, trlabel_raw

def load_cifar10_save(numdp, seed, flatten=True, target_class=None, **kwargs):

    # Data
    print('==> Preparing data..')

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    means = np.array([0.4914, 0.4822, 0.4465]).reshape([1,1,1,3])
    stds = np.array([0.2023, 0.1994, 0.2010]).reshape([1,1,1,3])

    train_data = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=True)

    test_data = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=True)

    if target_class is not None:
        data_0 = train_data.data[np.array(train_data.targets) == target_class[0]]
        data_1 = train_data.data[np.array(train_data.targets) == target_class[1]]
        targets = np.concatenate([np.zeros(data_0.shape[0]), np.ones(data_1.shape[0])])
        train_data.data = np.vstack([data_0, data_1])
        train_data.targets = targets.reshape(-1,1)
        test_data_0 = test_data.data[np.array(test_data.targets) == target_class[0]]
        test_data_1 = test_data.data[np.array(test_data.targets) == target_class[1]]
        test_targets = np.concatenate([np.zeros(test_data_0.shape[0]), np.ones(test_data_1.shape[0])])
        test_data.data = np.vstack([test_data_0, test_data_1])
        test_data.targets = test_targets.reshape(-1,1)

    num_train_samples = train_data.data.shape[0]
    local_rng = np.random.default_rng(seed=seed)
    sampled_idx = local_rng.choice(num_train_samples, numdp, replace=False)

    trdata_raw, trlabel_raw = copy.deepcopy(train_data.data[sampled_idx]), copy.deepcopy(np.array(train_data.targets)[sampled_idx])
    # if target_class is not None:
    #     trlabel_raw = (trlabel_raw == target_class[1]).astype(np.int32)
        
    trdata_raw, trlabel_raw = np.array(trdata_raw), np.array(trlabel_raw)
    
    trdata, trlabel, tedata, telabel = train_data.data/255, train_data.targets, test_data.data/255, test_data.targets
    trdata, tedata = (trdata - means)/stds, (tedata - means)/stds
    trdata, tedata = trdata.transpose([0,3,1,2]), tedata.transpose([0,3,1,2])
    if flatten:
        trdata, tedata = trdata.reshape(trdata.shape[0],-1), tedata.reshape(tedata.shape[0],-1)
    
    if target_class is None:
        trlabel, telabel = np.eye(10)[trlabel], np.eye(10)[telabel]

    # split test and train dataset
    trdata, trlabel = trdata[sampled_idx], trlabel[sampled_idx]
    trdata, tedata, trlabel, telabel = np.array(trdata), np.array(tedata), np.array(trlabel), np.array(telabel)
    
    # print(np.isnan(np.sum(trdata)), np.isfinite(np.sum(trdata)),np.isnan(np.sum(tedata)), np.isfinite(np.sum(tedata)))
    # raise NotImplementedError
    # raise KeyError

    return trdata, tedata, trlabel, telabel, trdata_raw, trlabel_raw 


def load_data(numdp=5000, dataset="rideshare", seed=43, flatten=True):
    trdata, tedata, trlabel, telabel = eval("load_"+dataset)(numdp=numdp, seed=seed, flatten=flatten)
    
    # print(f"Mean label {np.mean(telabel)}, Std label {np.std(telabel)}, Min label {np.min(telabel)}, Max label {np.max(telabel)}")
    # te_dis = pairwise_distances(tedata)
    # tr_dis = pairwise_distances(trdata)    
    # print(f"Test dataset, 5 quantile {np.quantile(te_dis, 0.05)}, 25 quantile {np.quantile(te_dis, 0.25)} median: {np.quantile(te_dis, 0.5)}, 70 quantile {np.quantile(te_dis, 0.7)}, 90 quantile {np.quantile(te_dis, 0.9)}, 95 quantile {np.quantile(te_dis, 0.95)}")
    # print(f"Train dataset, 5 quantile {np.quantile(tr_dis, 0.05)}, 25 quantile {np.quantile(tr_dis, 0.25)} median: {np.quantile(tr_dis, 0.5)}, 70 quantile {np.quantile(tr_dis, 0.7)}, 90 quantile {np.quantile(tr_dis, 0.9)}, 95 quantile {np.quantile(tr_dis, 0.95)}")
    # raise NotImplementedError
    
    return trdata, tedata, trlabel, telabel

def load_data_save(numdp=5000, dataset="rideshare", seed=43, flatten=True, target_class=None):
    trdata, tedata, trlabel, telabel, trdata_raw, trlabel_raw = eval("load_"+dataset+"_save")(numdp=numdp, seed=seed, flatten=flatten, target_class=target_class)
    
    # print(f"Mean label {np.mean(telabel)}, Std label {np.std(telabel)}, Min label {np.min(telabel)}, Max label {np.max(telabel)}")
    # te_dis = pairwise_distances(tedata)
    # tr_dis = pairwise_distances(trdata)    
    # print(f"Test dataset, 5 quantile {np.quantile(te_dis, 0.05)}, 25 quantile {np.quantile(te_dis, 0.25)} median: {np.quantile(te_dis, 0.5)}, 70 quantile {np.quantile(te_dis, 0.7)}, 90 quantile {np.quantile(te_dis, 0.9)}, 95 quantile {np.quantile(te_dis, 0.95)}")
    # print(f"Train dataset, 5 quantile {np.quantile(tr_dis, 0.05)}, 25 quantile {np.quantile(tr_dis, 0.25)} median: {np.quantile(tr_dis, 0.5)}, 70 quantile {np.quantile(tr_dis, 0.7)}, 90 quantile {np.quantile(tr_dis, 0.9)}, 95 quantile {np.quantile(tr_dis, 0.95)}")
    # raise NotImplementedError
    
    return trdata, tedata, trlabel, telabel, trdata_raw, trlabel_raw 