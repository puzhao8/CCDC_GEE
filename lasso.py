#%%
import pandas as pd

df_ = pd.read_csv("outputs\sampled_ts_data_WorldCover_stratified_V1_2626_2726.csv").set_index(['pnt_idx', 'seq_idx'])

not_nan_mask = df_['B12'].notna()
row_indices = not_nan_mask[not_nan_mask].index

pnt_idx_list = list(set([pnt_idx for pnt_idx, _ in list(row_indices)]))
# X_train = df_.loc[pnt_idx_list]

print(pnt_idx_list)


#%%

from sklearn.linear_model import LassoCV
import numpy as np

target_band = 'B12'
X = df_.loc[856][['FoY', target_band]]
# X = df_.loc[pnt_idx_list].reset_index()[['FoY', target_band]]

X['const'] = 1
X = X.rename(columns={'FoY': 't'})
X['cos2pt'] = X['t'].transform(lambda t: np.cos(2*np.pi*t))
X['sin2pt'] = X['t'].transform(lambda t: np.sin(2*np.pi*t))
X['cos4pt'] = X['t'].transform(lambda t: np.cos(4*np.pi*t))
X['sin4pt'] = X['t'].transform(lambda t: np.sin(4*np.pi*t))
X['cos6pt'] = X['t'].transform(lambda t: np.cos(6*np.pi*t))
X['sin6pt'] = X['t'].transform(lambda t: np.sin(6*np.pi*t))

X = X[['const', 't', 'cos2pt', 'sin2pt', 'cos4pt', 'sin4pt', 'cos6pt', 'sin6pt', target_band]]
X = X.dropna()
print(X.shape)

X_train = X[['const', 't', 'cos2pt', 'sin2pt', 'cos4pt', 'sin4pt', 'cos6pt', 'sin6pt']].values
y_train = X[[target_band]].values

# Assuming X_train and y_train are your features and target variable respectively
# X_train, y_train = np.random.rand(100, 10), np.random.rand(100)

# Create a LassoCV object
lasso_cv = LassoCV(cv=10, random_state=0, max_iter=10000)

# Fit model
lasso_cv.fit(X_train, y_train)

# Optimal lambda value
optimal_lambda = lasso_cv.alpha_
print("Optimal lambda (alpha) value:", optimal_lambda)



#%%
import numpy as np
import matplotlib.pyplot as plt

# t = np.linspace(2017, 2024, 219)
t = X['t'].values

def do_pred(t):
    A = lasso_cv.coef_
    X = np.array([1, t, np.cos(2*np.pi*t), np.sin(2*np.pi*t), np.cos(4*np.pi*t), np.sin(4*np.pi*t), np.cos(6*np.pi*t), np.cos(6*np.pi*t)])

    return ((X[:, np.newaxis].transpose()) @ (A[:, np.newaxis]))[0]

y = list(map(do_pred, list(t)))

plt.plot(t, X[target_band])
plt.plot(list(t), y)
plt.show()

