from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, SelectPercentile, RFECV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

def feat_select_model(model='svm'):
    if model == 'logistic':
        model = LogisticRegression()
    elif model == 'rf':
        model = RandomForestClassifier()
    else:
        model = SVC(kernel='linear')
    return model


def feat_filtering(X_train, y_train, X_test, method='percentile', k=10, func='chi2'):
    if func == 'f':
        f = f_classif
    elif func == 'mutual_info':
        f = mutual_info_classif
    else:
        f = chi2
    if method == 'k':
        sel = SelectKBest(score_func=f, k=k)
    else:
        sel = SelectPercentile(score_func=f, percentile=k)
    
    X_train_redux = sel.fit_transform(X_train, y_train)
    X_test_redux = sel.transform(X_test)
    return sel, X_train_redux, X_test_redux

def rfe(X_train, y_train, X_test, step=0.1, min_features_to_select=1, cv=5, model='svm', save_file=False):
    _model = feat_select_model(model)
    sel = RFECV(
            estimator=_model,
            step=step,
            cv=cv,
            scoring="accuracy",
            min_features_to_select=min_features_to_select,
            n_jobs=2,
        )
    X_train_redux = sel.fit_transform(X_train, y_train)
    X_test_redux = sel.transform(X_test)

    if save_file:
        with open("rfecv_" + model + ".pickle", "wb") as fh:
            pickle.dump(sel, fh)

    return sel, X_train_redux, X_test_redux