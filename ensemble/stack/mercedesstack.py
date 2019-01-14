import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD

# Load the data
print("load the data ...")
train = pd.read_csv('./MercedesBenzGreenerManufacturing/train.csv')
test = pd.read_csv('./MercedesBenzGreenerManufacturing/test.csv')

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

print("pandas get_dummies ...")

df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

print("train head ...")
print(train.head())

print("test head ...")
print(test.head())

# if False:
n_comp = 12

# tSVD
print("TruncatedSVD ...")
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

# PCA
print("PCA ...")
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)

# ICA
print("FastICA ...")
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

# GRP
print("GaussianRandomProjection ...")
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

# SRP
print("SparseRandomProjection ...")
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

class StackingCVRegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        for i, clf in enumerate(self.regressors):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)

                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
                np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)


class StackingCVRegressorRetrained(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5, use_features_in_secondary=False):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        # Create out-of-fold predictions for training meta-model
        for i, regr in enumerate(self.regr_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(regr)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])

        # Train meta-model
        if self.use_features_in_secondary:
            self.meta_regr_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_regr_.fit(out_of_fold_predictions, y)

        # Retrain base models on all data
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])

        if self.use_features_in_secondary:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(meta_features)

class StackingEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]

        # Train base models
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        return np.mean(predictions, axis=1)

en = make_pipeline(RobustScaler(), SelectFromModel(Lasso(alpha=0.03)), ElasticNet(alpha=0.001, l1_ratio=0.1))

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()
)


# stacked_pipeline.fit(finaltrainset, y_train)
# results = stacked_pipeline.predict(finaltestset)

rf = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)

et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)

xgbm = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.9, base_score=y_mean,
                                objective='reg:linear', n_estimators=1000)

stack_avg = StackingCVRegressorAveraged((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

stack_with_feats = StackingCVRegressorRetrained((en, rf, et), xgbm, use_features_in_secondary=True)

stack_with_feats2 = StackingCVRegressorRetrained((en, rf, et), xgbm, use_features_in_secondary=False)

stack_retrain = StackingCVRegressorRetrained((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

averaged = AveragingRegressor((en, rf, et, xgbm))
if False:
    results = cross_val_score(en, train.values, y_train, cv=5, scoring='r2')
    print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(stacked_pipeline, train.values, y_train, cv=5, scoring='r2')
    print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(rf, train.values, y_train, cv=5, scoring='r2')
    print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(et, train.values, y_train, cv=5, scoring='r2')
    print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(xgbm, train.values, y_train, cv=5, scoring='r2')
    print("XGBoost score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(averaged, train.values, y_train, cv=5, scoring='r2')
    print("Averaged base models score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(stack_with_feats, train.values, y_train, cv=5, scoring='r2')
    print("Stacking (with primary second feats) score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(stack_with_feats2, train.values, y_train, cv=5, scoring='r2')
    print("Stacking (with primary feats) score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(stack_retrain, train.values, y_train, cv=5, scoring='r2')
    print("Stacking (retrained) score: %.4f (%.4f)" % (results.mean(), results.std()))

    results = cross_val_score(stack_avg, train.values, y_train, cv=5, scoring='r2')
    print("Stacking (averaged) score: %.4f (%.4f)" % (results.mean(), results.std()))


'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 0
}
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1250

# train model
xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = xgb_model.predict(dtest)

import matplotlib.pyplot as plt
# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(xgb_model, max_num_features=50, height=0.8, ax=ax)
plt.show()

# Perform cross-validation: cv_results
cv_results = xgb.cv(params=dict(xgb_params, silent=0), dtrain=dtrain, nfold=5, num_boost_round=num_boost_rounds, metrics=['rmse', 'error'], as_pandas=True, seed=123, early_stopping_rounds=25)

# Print cv_results
print(cv_results)
if False:
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(train, y_train, random_state=16)

    from keras import optimizers
    adam = optimizers.adam(lr=0.0001)

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers.normalization import BatchNormalization

    def create_model():
        model = Sequential()
        model.add(Dense(500, input_dim=631, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        model.add(Dense(300, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        model.add(Dense(150, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        model.add(Dense(75, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        model.add(Dense(1, kernel_initializer='normal', activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

    model = create_model()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, verbose=0, epochs=100)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Success rate: %.2f%%" % (score[1]*100))
    print("Random guess 1-20 bins: %.2f%%" % (1/20*100))
