
#%%
import pandas as pd
from pathlib import Path
import numpy as np
from group_bands import feature_dict, label_bands, fused_class_dict

np.random.seed(42)

# configuration
version = 'V1'
y_col = 'wetland_mask'
band_select = True

df = pd.read_csv(f"data/training/sampled_points_with_fused_label_stratified_1k_per_cls_V1.csv")
df.loc[df.wetland_mask > 1, 'wetland_mask'] = 0
 

save_dir = Path(f"outputs/feature_importance_{version}/{y_col}_TEST")
save_dir.mkdir(exist_ok=True, parents=True)

# # for feature_key in feature_dict.keys():
# for feature_key in ['TOPO', 'S1', 'S2', 'SS', 'ALL', 'S1_coefs', 'S2_coefs', 'SS_coefs', 'S1_HM', 'S2_HM', 'SS_HM']:
#     print(f"------------------- {feature_key} -------------------------")

# remove some label bands 
rmse_bands = list(df.filter(regex='.*rmse.*').columns)
drop_bands = label_bands + ['idx', 'geometry', 'lat', 'lon']
drop_bands += rmse_bands

if band_select: # select a few bands
    # selected_bands = feature_dict[feature_key]
    # print(f"feature_key: {feature_key}")
    selected_bands = ['rem', 'elevation', 'slope', 'twi']
    X = df[selected_bands + [y_col]]

else:
    drop_bands.remove(y_col)
    X = df.drop(drop_bands, axis=1)

X = X.dropna()
X.replace([np.inf, -np.inf], 0, inplace=True)

y = X[y_col].astype(int)

# # class names 
# unique_label_idx = sorted(y.unique())
# cls_name_bank = list(fused_class_dict.values())
# cls_names = [cls_name_bank[cls] for cls in unique_label_idx]
# print(f"unique_label_idx: {unique_label_idx}")

# remove labels from X
X = X.drop([y_col], axis=1) 
num_bands = X.shape[-1]

print(f"{y_col} ({num_bands} bands)")
print(X.shape, y.shape)
print(list(X.columns))

#%%
# referenece: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier 

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image


NUM_of_TREES = 2

# from sklearn.datasets import load_breast_cancer
# X, y = load_breast_cancer(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=NUM_of_TREES, max_depth=5, random_state=42)
# clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Baseline accuracy on train data: {train_acc:.2}")
print(f"Baseline accuracy on test data: {test_acc:.2}")

# report on test accuracy
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print(df_report)
# df_report.to_csv(save_dir / f'{feature_key}_{y_col}_{num_bands}_bands_AccReport.csv')


#%%

def save_tree_as_png(model, tree_index, feature_names, target_names, save_dir):
    tree_name = f"tree_{tree_index}"
    # Extract single tree
    estimator = model.estimators_[tree_index]

    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(estimator, out_file=str(save_dir / f'{tree_name}.dot'), 
                    feature_names = feature_names,
                    class_names = target_names,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    # call(['dot', '-Tpng', str(save_dir / f'{tree_name}.dot'), '-o', str(save_dir /f'{tree_name}.png'), '-Gdpi=600'])
    call(['dot', '-Tpdf', str(save_dir / f'{tree_name}.dot'), '-o', str(save_dir /f'{tree_name}.pdf'), '-Gdpi=600'])

    # # Display in jupyter notebook
    # from IPython.display import Image
    # Image(filename = f'{tree_name}.png')


for tree_index in range(NUM_of_TREES):
    print(tree_index)
    save_tree_as_png(clf, tree_index, selected_bands, ['non-wetland', 'wetland'], save_dir)

#%%

from IPython.display import Image

tree_index = 2
Image(filename = save_dir / f'tree_{tree_index}.png')

#%%
from sklearn import tree

for tree_index in range(NUM_of_TREES):
  text_representation = tree.export_text(decision_tree=clf.estimators_[tree_index], 
                                         feature_names=selected_bands,
                                         class_names=['non-wetland', 'wetland'])
  with open(save_dir / f'tree_{tree_index}.txt', 'w') as f:
    f.write(text_representation)
    f.close()

    print(text_representation)

#%%
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=selected_bands,  
                   class_names=['non-wetland', 'wetland'],
                   filled=True)
fig.savefig(save_dir / "decistion_tree.png")


#%%
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=selected_bands,  
                                class_names=['non-wetland', 'wetland'],
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph