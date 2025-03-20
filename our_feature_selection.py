from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, SelectPercentile

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