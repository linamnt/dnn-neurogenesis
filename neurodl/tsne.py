import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

def tsne_preprocess(middle, targets, shape=1000):
    df = pd.DataFrame(index=range(len(middle)), columns=list(range(shape)) + ["Label"])
    df.iloc[:, :shape] = middle.detach().numpy()
    df.loc[:, "Label"] = targets.numpy().astype(int)
    return df


def tsne_transform(df, shape=1000):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    embedding = tsne.fit_transform(df.iloc[:, :shape].values)
    return tsne, embedding


def tsne_plot(df, embedding, to_file=None):
    df_copy = df.copy()
    df_copy["x"] = embedding[:, 0]
    df_copy["y"] = embedding[:, 1]

    fig = sns.lmplot(
        x="x",
        y="y",
        data=df_copy,
        hue="Label",
        fit_reg=False,
        scatter_kws={"alpha": 0.3, "s": 5},
    )
    if to_file is not None:
        plt.savefig(to_file, dpi=300)
  


def tsne_pipe(middle, targets, plot=True, shape=1000, to_file=None):
    df = tsne_preprocess(middle, targets, shape)
    tsne, embedding = tsne_transform(df, shape)
    kld = tsne.kl_divergence_
    tsne_plot(df, embedding, to_file)
    targets = targets.squeeze()
    silhouette = silhouette_score(embedding, targets)
    return kld, silhouette
