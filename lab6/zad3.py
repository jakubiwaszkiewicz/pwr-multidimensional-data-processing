import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
hsi_small = np.load('./hsi_small.npy')

fig, ax = plt.subplots(1, 2)


r = hsi_small[:,:,100:101]
g = hsi_small[:,:,50:51]
b = hsi_small[:,:,7:8]


r -= np.min(r)
r /= np.max(r)

g -= np.min(g)
g /= np.max(g)

b -= np.min(b)
b /= np.max(b)

rgb = np.dstack((r,g,b))

prepared_hsi = hsi_small.reshape(65536, 204)
pca = PCA(n_components=3)
pca_transformed_hsi = pca.fit_transform(prepared_hsi)
img_pca = pca_transformed_hsi.reshape(256,256,3)

dataset_rgb = rgb.reshape(65536,3)
dataset_pca = pca_transformed_hsi
dataset_all = hsi_small.reshape(65536,204)
y = np.load('./hsi_small_gt.npy')

clfs = [
    GaussianNB(),
    SVC(),
    MLPClassifier(),
]


for clf in clfs:
    for jdx, X in enumerate([dataset_rgb, dataset_pca, dataset_all]):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
        y = y.flatten()
        results = []
        for idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
            clf.fit(X[train_index], y[train_index])
            y_pred = clf.predict(X[test_index])
            results.append(balanced_accuracy_score(y[test_index], y_pred))
        std_res = np.std(results)
        mean_res = np.mean(results)
        print(f"Clf: {clf}")
        print(f"Dataset {jdx}")
        print(f"Accuracy: {round(mean_res, 3)}")
        print(f"Standard Deviation: {round(std_res, 3)}")
