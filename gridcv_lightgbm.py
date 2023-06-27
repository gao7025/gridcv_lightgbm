# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib

import warnings
warnings.filterwarnings("ignore")


class TrainCV:
    def __int__(self):
        pass

    params_tot = {}
    '''
    参数初始值,简单看下效果
    '''
    def initial_params(self, x, y):
        print('---参数初始值,简单看下效果----')
        warnings.filterwarnings("ignore")
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves': 50,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'force_col_wise': True
        }
        data_train = lgb.Dataset(x, y, silent=True)
        cv_results = lgb.cv(
            params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
            early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
        print('-------------', cv_results)
        print('best n_estimators:', len(cv_results['auc-mean']))
        print('best cv score:', cv_results['auc-mean'][-1])
        self.params_tot.update({'best_n_estimator': len(cv_results['auc-mean'])})
        return len(cv_results['auc-mean'])  # cv_results['auc-mean'][-1]
    '''
    这是提高精确度的最重要的参数。
    `max_depth` ：设置树深度，深度越大可能过拟合
    `num_leaves`：因为 LightGBM 使用的是 leaf-wise 的算法，因此在调节树的复杂程度时，使用的是 num_leaves 而不是 max_depth。
    大致换算关系：num_leaves = 2^(max_depth)，但是它的值的设置应该小于 2^(max_depth)，否则可能会导致过拟合。
    对于这两个参数调优，我们先粗调，再细调，sklearn模型评估里的scoring参数都是采用的
    **higher return values are better than lower return values（较高的返回值优于较低的返回值）**。
    但是，我采用的metric策略采用的是auc，越高越好。
    '''
    def get_depth_leaves(self, x, y):
        warnings.filterwarnings("ignore")
        best_n_estimator = self.initial_params(x, y)
        # 我们可以创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目)
        model_lgb = lgb.LGBMClassifier(objective='binary', num_leaves=50,
                                       learning_rate=0.1, n_estimators=best_n_estimator, max_depth=6,
                                       metric='auc', bagging_fraction=0.8, feature_fraction=0.8)
        params_test1 = {
            'max_depth': range(3, 9, 2),
            'num_leaves': range(50, 150, 20)
        }
        gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='roc_auc', cv=5,
                                verbose=-1, n_jobs=4)
        gsearch1.fit(x, y)
        # means_train = gsearch1.cv_results_['mean_train_score']
        # std_train = gsearch1.cv_results_['std_train_score']
        means = gsearch1.cv_results_['mean_test_score']
        std = gsearch1.cv_results_['std_test_score']
        params = gsearch1.cv_results_['params']
        for mean, std, param in zip(means, std, params):
            print("mean : %f std : %f %r" % (mean, std, param))
        print('best_params :', gsearch1.best_params_, gsearch1.best_score_)
        best_max_depth = gsearch1.best_params_.get('max_depth')
        best_num_leaves = gsearch1.best_params_.get('num_leaves')
        new_params = {
            'best_max_depth': best_max_depth,
            'best_num_leaves': best_num_leaves
        }
        self.params_tot.update(new_params)

        return best_n_estimator, best_max_depth, best_num_leaves
    '''
    说到这里，就该降低过拟合了。
    `min_data_in_leaf` 是一个很重要的参数, 也叫min_child_samples，它的值取决于训练数据的样本个树和num_leaves. 
    将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合。
    `min_sum_hessian_in_leaf`：也叫min_child_weight，使一个结点分裂的最小海森值之和，
    Minimum sum of hessians in one leaf to allow a split. Higher values potentially decrease over fitting）
    我们采用跟上面相同的方法进行：
    '''
    def min_leaf(self, x, y):

        warnings.filterwarnings("ignore")
        print('---树深和叶子结点树训练----')
        best_n_estimator, best_max_depth, best_num_leaves = self.get_depth_leaves(x, y)
        params_test3 = {
            'min_child_samples': [18, 19, 20, 21, 22],
            'min_child_weight': [0.001, 0.002]
        }
        model_lgb3 = lgb.LGBMClassifier(objective='binary', num_leaves=best_num_leaves, learning_rate=0.1,
                                        n_estimators=best_n_estimator, max_depth=best_max_depth,
                                        metric='auc', bagging_fraction=0.8, feature_fraction=0.8)
        gsearch3 = GridSearchCV(estimator=model_lgb3, param_grid=params_test3, scoring='roc_auc', cv=5,
                                verbose=-1, n_jobs=4)
        gsearch3.fit(x, y)
        means = gsearch3.cv_results_['mean_test_score']
        std = gsearch3.cv_results_['std_test_score']
        params = gsearch3.cv_results_['params']
        for mean, std, param in zip(means, std, params):
            print("mean : %f std : %f %r" % (mean, std, param))
        print('best_params :', gsearch3.best_params_, gsearch3.best_score_)
        best_min_child_samples = gsearch3.best_params_.get('min_child_samples')
        best_min_child_weight = gsearch3.best_params_.get('min_child_weight')
        new_params = {
            'best_min_child_samples': best_min_child_samples,
            'best_min_child_weight': best_min_child_weight
        }
        self.params_tot.update(new_params)
        return best_min_child_samples, best_min_child_weight
    '''
    这两个参数都是为了降低过拟合的。
    feature_fraction参数来进行特征的子抽样。这个参数可以用来防止过拟合及提高训练速度。
    bagging_fraction+bagging_freq参数必须同时设置，bagging_fraction相当于subsample样本采样，可以使bagging更快的运行，同时也可以降拟合。bagging_freq默认0，表示bagging的频率，0意味着没有使用bagging，k意味着每k轮迭代进行一次bagging。
    不同的参数，同样的方法。
    '''
    def get_fraction(self, x, y):

        warnings.filterwarnings("ignore")
        # best_n_estimator, best_max_depth, best_num_leaves = self.get_depth_leaves(x, y)
        print('----叶子结点最小数量和最小 hessian和的训练----')
        best_min_child_samples, best_min_child_weight = self.min_leaf(x, y)
        params_test4 = {
            'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
            'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        best_n_estimator = self.params_tot.get('best_n_estimator')
        best_max_depth = self.params_tot.get('best_max_depth')
        best_num_leaves = self.params_tot.get('best_num_leaves')
        model_lgb4 = lgb.LGBMClassifier(objective='binary',
                                        learning_rate=0.1, n_estimators=best_n_estimator,
                                        max_depth=best_max_depth, num_leaves=best_num_leaves,
                                        min_child_samples=best_min_child_samples,
                                        min_child_weight=best_min_child_weight,
                                        metric='auc', bagging_fraction=0.8, feature_fraction=0.8)
        gsearch4 = GridSearchCV(estimator=model_lgb4, param_grid=params_test4, scoring='roc_auc', cv=5,
                                verbose=-1, n_jobs=4)
        gsearch4.fit(x, y)
        means = gsearch4.cv_results_['mean_test_score']
        std = gsearch4.cv_results_['std_test_score']
        params = gsearch4.cv_results_['params']
        for mean, std, param in zip(means, std, params):
            print("mean : %f std : %f %r" % (mean, std, param))
        print('best_params :', gsearch4.best_params_, gsearch4.best_score_)
        best_feature_fraction = gsearch4.best_params_.get('feature_fraction')
        best_bagging_fraction = gsearch4.best_params_.get('bagging_fraction')
        new_params = {
            'best_feature_fraction': best_feature_fraction,
            'best_bagging_fraction': best_bagging_fraction
        }
        self.params_tot.update(new_params)
        print(self.params_tot)
        return best_feature_fraction, best_bagging_fraction
    '''
    正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)，毫无疑问，是降低过拟合的，两者分别对应l1正则化和l2正则化。我们也来尝试一下使用这两个参数。
    '''
    def get_alpha_lambda(self, x, y):

        warnings.filterwarnings("ignore")
        print('----特征抽样和bagging抽样训练----')
        best_feature_fraction, best_bagging_fraction = self.get_fraction(x, y)
        params_test6 = {
            'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
            'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
        }
        best_n_estimator = self.params_tot.get('best_n_estimator')
        best_max_depth = self.params_tot.get('best_max_depth')
        best_num_leaves = self.params_tot.get('best_num_leaves')
        best_min_child_samples = self.params_tot.get('best_min_child_samples')
        best_min_child_weight = self.params_tot.get('best_min_child_weight')
        model_lgb6 = lgb.LGBMClassifier(objective='binary',
                                        learning_rate=0.01, n_estimators=best_n_estimator,
                                        max_depth=best_max_depth, num_leaves=best_num_leaves,
                                        min_child_samples=best_min_child_samples,
                                        min_child_weight=best_min_child_weight,
                                        feature_fraction=best_feature_fraction, bagging_fraction=best_bagging_fraction,
                                        metric='auc')
        gsearch6 = GridSearchCV(estimator=model_lgb6, param_grid=params_test6, scoring='roc_auc', cv=5,
                                verbose=-1, n_jobs=4)
        gsearch6.fit(x, y)
        means = gsearch6.cv_results_['mean_test_score']
        std = gsearch6.cv_results_['std_test_score']
        params = gsearch6.cv_results_['params']
        for mean, std, param in zip(means, std, params):
            print("mean : %f std : %f %r" % (mean, std, param))
        print('best_params :', gsearch6.best_params_, gsearch6.best_score_)
        best_reg_alpha = gsearch6.best_params_.get('reg_alpha')
        best_reg_lambda = gsearch6.best_params_.get('reg_lambda')
        new_params = {
            'best_reg_alpha': best_reg_alpha,
            'best_reg_lambda': best_reg_lambda
        }
        self.params_tot.update(new_params)
        print(self.params_tot)
        return best_reg_alpha, best_reg_lambda

    '''
    降低learning_rate**
    之前使用较高的学习速率是因为可以让收敛更快，但是准确度肯定没有细水长流来的好。最后，我们使用较低的学习速率，以及使用更多的决策树n_estimators来训练数据，看能不能可以进一步的优化分数。
    我们可以用回lightGBM的cv函数了 ，我们代入之前优化好的参数。
    '''
    def train_lgb(self, x, y, x_test, y_test):

        warnings.filterwarnings("ignore")
        print('----l1和l2正则训练----')
        best_reg_alpha, best_reg_lambda = self.get_alpha_lambda(x, y)
        print('----获取之前调优的最佳参数----')
        best_n_estimator = self.params_tot.get('best_n_estimator')
        best_max_depth = self.params_tot.get('best_max_depth')
        best_num_leaves = self.params_tot.get('best_num_leaves')
        best_min_child_samples = self.params_tot.get('best_min_child_samples')
        best_min_child_weight = self.params_tot.get('best_min_child_weight')
        best_feature_fraction = self.params_tot.get('best_feature_fraction')
        best_bagging_fraction = self.params_tot.get('best_bagging_fraction')
        print('----降低学习率并用之前最有参数训练----')
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.01,
            'n_estimator': best_n_estimator,
            'num_leaves': best_num_leaves,
            'max_depth': best_max_depth,
            'min_data_in_leaf': best_min_child_samples,
            'min_sum_hessian_in_leaf': best_min_child_weight,
            'lambda_l1': best_reg_alpha,
            'lambda_l2': best_reg_lambda,
            'feature_fraction': best_feature_fraction,
            'bagging_fraction': best_bagging_fraction
        }
        data_train = lgb.Dataset(x, y, silent=True)
        cv_results = lgb.cv(
            params, data_train,  # num_boost_round=1000,
            nfold=5, stratified=False, shuffle=True, metrics='auc',
            early_stopping_rounds=100, verbose_eval=100, show_stdv=True)
        print('best cv score:', cv_results['auc-mean'][-1])
        print('best params:', self.params_tot)
        # 重新定义模型并传入已调好的参数
        print('----重新定义模型传入最终参数并保存模型----')
        lgb_train = lgb.Dataset(x, y)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
        eval_result = {}
        lgb_model = lgb.train(params, lgb_train,  # num_boost_round=1000,
                              valid_sets=lgb_eval, evals_result=eval_result,
                              early_stopping_rounds=100)
        import joblib
        joblib.dump(lgb_model, 'lGbmModel_1024.pkl')

        return self.params_tot
