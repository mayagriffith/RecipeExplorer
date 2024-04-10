import warnings

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import cluster_search

# sklearn has a lot of unhelpful warnings, so filter those out.
warnings.filterwarnings('ignore')
all_recipe_data = cluster_search.get_df_all_features()
only_macro_data = cluster_search.get_df_macros(all_recipe_data)
macro_ratios_df = cluster_search.get_macro_ratios(only_macro_data)

# Define all the feature sets to try. Drop the title column, since KMeans
# can't process strings.
feature_sets = {
    'all_recipe_data': all_recipe_data.drop('title', axis=1),
    'only_macro_data': only_macro_data.drop('title', axis=1),
    'only_macro_ratios': macro_ratios_df,
    'all_plus_ratios': all_recipe_data.join(
        macro_ratios_df).drop('title', axis=1)
}

CLUSTERS_RANGE = range(2, 8)

# Data structure to keep track of inputs, outputs, and analysis metadata for
# one possible configuration of the KMeans algorithm to use for recommending
# recipes to users.
class KMeansCandidate:
    def __init__(self, fs_name, df, num_clusters):
        self.fs_name = fs_name
        self.df = df
        self.num_clusters = num_clusters
        self.model = KMeans(num_clusters)
        self.labels = self.model.fit_predict(df)
        self.sil_score = silhouette_score(df, self.labels)

    # Visualize the reduced dimension data frame using this candidate's cluster labels
    def show_clusters(self, reduced_df):
        plt.figure()
        for i in range(self.num_clusters):
            tsne_cluster = reduced_df[self.labels == i]
            plt.scatter(tsne_cluster[:,0], tsne_cluster[:,1], label=f"Cluster {i+1}", s=10)
        plt.title(f'Feature set {self.fs_name} with {self.num_clusters} clusters')
        plt.legend(title="Cluster Labels")
        plt.tight_layout()
        plt.savefig(f'clusters_{self.fs_name}_{self.num_clusters}.png')
        plt.show()
        plt.close()

    def fit_user_by_macros(self):
        # TODO: Figure out which cluster best fits a user's macro preferences,
        # and how good that fit is quantitatively.
        best_label = 'TODO'
        best_fit_score = 'TODO'
        return best_label, best_fit_score

    def fit_user_by_prefs(self):
        # TODO: Figure out which cluster best fits a user's macro preferences,
        # and how good that fit is quantitatively.
        best_label = 'TODO'
        best_fit_score = 'TODO'
        return best_label, best_fit_score


# Try running KMeans on the given data for a variety of cluster sizes.
def sweep_cluster_size(fs_name, df):
    candidates = []
    max_sil_score = 0.0
    print('silhouette scores: ', end='', flush=True)
    for num_clusters in CLUSTERS_RANGE:
        candidate = KMeansCandidate(fs_name, df, num_clusters)
        candidates.append(candidate)
        print(f'{num_clusters}: {candidate.sil_score:5.3f}, ',
              end='', flush=True)
        if candidate.sil_score > max_sil_score:
            max_sil_score = candidate.sil_score
    print()
    return candidates, max_sil_score


# Do further analysis to visualize the quality of categories discovered for a
# particular set of features. Considers only cluster counts that produced a
# silhouette score close to the best for this feature set.
def evaluate_candidates(candidates, max_sil_score, reduced_df):
    candidates = [
        candidate for candidate in candidates
        if candidate.sil_score >= .9 * max_sil_score]
    print(f'Doing further evaluation on {len(candidates)} candidates...')
    for candidate in candidates:
        candidate.show_clusters(reduced_df)
        _, best_fit_macros = candidate.fit_user_by_macros()
        _, best_fit_prefs = candidate.fit_user_by_prefs()
        print(f'{candidate.num_clusters} clusters -> '
              f'sil:{candidate.sil_score:5.3f}, '
              f'macros: {best_fit_macros}, prefs: {best_fit_prefs}')
        _, counts = np.unique(candidate.labels, return_counts=True)
        print(f'cluster sizes: {counts}')
    print()


# Project the data in df into two dimensions for visualization.
def reduce(df):
    # standardize
    df = StandardScaler().fit_transform(df)

    if df.shape[1] > 50:
        # Reduce feature size to 50 (recommended pre-processing for running
        # TSNE on large datasets)
        df = PCA(n_components=50).fit_transform(df)

    # Reduce to two components using TSNE for visualization.
    return TSNE(random_state=42).fit_transform(df)


# Try running KMeans with several different variations of our data set with
# different features, and different numbers of desired clusters to try to find
# a configuration that allows us to partition the data to suit our users.
#
# The results of this suggest that KMeans may be a poor choice of algorithm for
# this task, and that feature selection won't help.
#
# Before, we just used macro ratios as the features. KMeans happily divided the
# feature set into N roughly equally-sized categories, but the category
# divisions didn't seem very meaningful. When visualizing the data flattened
# with TSNE, it looked like totally arbitrary cuts that didn't correlate with
# the data's structure, and the silhouette scores were quite bad. I'm not sure
# what this means, but my guess is just that the recipe data itself my not be
# cleanly separable in the way we want. There may not be well-isolated groups
# of recipes with similar nutiritional values.
#
# So, I tried adding in the raw macro scores and the ingredient / "tag"
# features, as well. When I do that, I get much better silhouette scores, but
# KMeans produces highly imbalanced clusters, putting 98+% of elements into one
# bucket. I guess the good silhouette scores just reflect putting the outliers
# into their own cluster. Doing some quick searches online, imbalanced clusters
# are apparently common when running KMeans on sparse, non-continuous data such
# as binary or categorical data... which is, of course, what I added. So, I
# don't think these extra features help at all.
def main():
    for fs_name, df in feature_sets.items():
        print(f'Evaluating feature set: {fs_name}')
        candidates, max_sil_score = sweep_cluster_size(fs_name, df)
        evaluate_candidates(candidates, max_sil_score, reduce(df))

if __name__ == '__main__':
    main()
