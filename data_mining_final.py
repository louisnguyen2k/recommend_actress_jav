import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from google_trans_new import google_translator
from datetime import date
import warnings
warnings.filterwarnings("ignore")


actress = pd.read_csv('actress_clean.csv')

# AVG data
#
AVG_HEIGHT = actress.height.mean()
AVG_BUST = actress.bust.mean()
AVG_WAIST = actress.waist.mean()
AVG_HIPS = actress.hips.mean()

df = actress[['height', 'bust', 'waist', 'hips']]
actress_np = df.to_numpy()
# print(actress_np)

# elbow trick
#


def elbow_trick(data_np, K_limit):
    sum_distance = []
    K = range(1, K_limit)
    for k in K:
        k_mean = KMeans(n_clusters=k)
        k_mean.fit(data_np)
        sum_distance.append(k_mean.inertia_)
    return K, sum_distance


K, sum_distance = elbow_trick(actress_np, K_limit=10)
plt.plot(K, sum_distance, 'bx-')
plt.show()


# metrics try
#
def cal_kmeans_metrics_score(data_np, min_K, max_K):
    kmeans_metrics_ = []
    k_optimal = None
    K_silhouette_max_val = 0
    K_calinski_max_val = 0
    for i in range(min_K, max_K+1):
        k_mean_ = KMeans(n_clusters=i)
        model = k_mean_.fit(data_np)
        label_ = k_mean_.labels_
        metrics_silhouette_score_ = metrics.silhouette_score(
            data_np, label_, metric='euclidean')
        metrics_calinski_ = metrics.calinski_harabasz_score(data_np, label_)
        #
        res_dict_ = {
            'n_clusters': i,
            'k_mean_': k_mean_,
            'model': model,
            'label': label_,
            'metrics_silhouette_score_': metrics_silhouette_score_,
            'metrics_calinski_': metrics_calinski_,
        }
        kmeans_metrics_.append(res_dict_)
        #
        if K_silhouette_max_val < metrics_silhouette_score_ and K_calinski_max_val < metrics_calinski_:
            k_optimal = res_dict_
            K_silhouette_max_val = metrics_silhouette_score_
            K_calinski_max_val = metrics_calinski_
    return k_optimal, kmeans_metrics_


k_optimal, kmeans_metrics_ = cal_kmeans_metrics_score(
    actress_np, min_K=2, max_K=5)
# for kmean_metric in kmeans_metrics_:
#     print('kmean_metric', kmean_metric)
# print('k_optimal', k_optimal)

# visualize kmeans with plt
#


def visualize_Kmeans(data_np, model, label, k):
    color = ['green', 'blue', 'orange']
    for i in range(k):
        plt.scatter(
            data_np[label == i, 0],
            data_np[label == i, 1],
            label='cluster ' + str(i),
            c=color[i],
            edgecolors='black'
        )
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker='*', s=250,
        c='red',
        label='centroids',
        edgecolors='black'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()


visualize_Kmeans(actress_np, k_optimal['model'],
                 k_optimal['label'], k_optimal['n_clusters'])


# input: age, height, cup_size, bust, waist, hips
# => output: which cluster?
# random(10) in clusters
#
df1 = actress[['id', 'height', 'bust', 'waist', 'hips']]
# print(df1.info())

df2 = actress[['id', 'name', 'imgurl', 'birthplace', 'hobby', 'cup_size']]
# print(df2)


lookup = df1.merge(df2, on='id', how='left')
# print(lookup.info())
lookup['cluster'] = k_optimal['label']
# print(lookup)

# recommend actress function
#


def recommend(model, height, bust, waist, hips):
    arr = np.array([[height, bust, waist, hips]])
    pred = model.predict(arr)
    res = lookup[lookup['cluster'] == pred[0]].sample(10)
    # translator = google_translator()
    # res['name_english'] = res['name'].apply(translator.translate, lang_src='ja', lang_tgt='en')
    # res['hobby_english'] = res['hobby'].apply(translator.translate, lang_src='ja', lang_tgt='en')
    return res


# Input data, you can change input here
#
model = k_optimal['model']
height = None or AVG_HEIGHT
bust = None or AVG_BUST
waist = None or AVG_WAIST
hips = None or AVG_HIPS
res = recommend(model, height, bust, waist, hips)
print(res)
