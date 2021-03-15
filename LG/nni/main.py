import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import *
import joblib
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import nni

PREDICT_PATH = "../nnitest"

def load_data():
    train_data = pd.read_csv('../LG_data/train_x.csv')
    test_data = pd.read_csv('../LG_data/test_x.csv')
    target = train_data['problem']
    train_data.drop(['problem'], axis=1, inplace=True)

    return train_data, target, test_data

# validation auc score를 확인하기 위해 정의
def f_pr_auc(y_pred, y_true):
    fpr, tpr, _  = roc_curve(y_true, y_pred)
    score = auc(fpr, tpr)
    return score

def get_default_parameters():
    params = {
        'max_depth': 10,
        'num_leaves': 20,
        'learning_rate': 0.1,
        'subsample': 1,
        'subsample_freq': 0,
    }
    return params

def make_submission(y_pred, eval_score):

    pass

def train(data,target,params):
    n_splits = 5
    kford = KFold(n_splits=n_splits, random_state=2020, shuffle = True)
    val_scores = []
    oof_test_pred = np.zeros(14999)

    print('Start training...')
    for train_idx, valid_idx in kford.split(data, target):
        x_train, y_train = data.iloc[train_idx], target.iloc[train_idx]
        x_valid, y_valid = data.iloc[valid_idx], target.iloc[valid_idx]

        lgbm = LGBMClassifier(n_estimators=20000, random_state=2020,
                             # device = 'gpu',
                             num_leaves=params['num_leaves'],
                             max_depth=params['max_depth'],
                             learning_rate=params['learning_rate'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             reg_alpha=params['reg_alpha'],
                             n_jobs=-1,
                             )
        lgbm.fit(x_train, y_train,
                 eval_set=[(x_valid, y_valid)],
                 eval_metric='auc',
                 early_stopping_rounds=1000,
                 verbose=100)

        y_valid_pred = lgbm.predict(x_valid)

        # 예측결과는 5로 나누어 더함.
        y_test_pred = lgbm.predict(test_data)
        oof_test_pred += y_test_pred / 5

        valid_auc = f_pr_auc(y_valid_pred, y_valid)
        val_scores.append(valid_auc)

        y_valid_pred = lgbm.predict(x_valid)
        valid_auc = f_pr_auc(y_valid_pred,y_valid)
        val_scores.append(valid_auc)

    eval_score = np.mean(val_scores)
    print('The auc of prediction is:', eval_score)
    nni.report_final_result(eval_score)


if __name__ == '__main__':
    data , target, test = load_data()

    RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = get_default_parameters()

    PARAMS.update(RECEIVED_PARAMS)

    #train
    train(data, target, PARAMS)