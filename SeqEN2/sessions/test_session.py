#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from datetime import datetime
from os import system
from os.path import dirname
from pathlib import Path

import plotly.express as px
from numpy import array, unique
from pandas import concat
from plotly.offline import plot
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding

from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import read_json
from SeqEN2.model.model import Model
from SeqEN2.utils.custom_arg_parser import TestSessionArgParser


class TestSession:

    root = Path(dirname(__file__)).parent.parent

    def __init__(self):
        # setup dirs
        self.models_dir = self.root / "models"
        if not self.models_dir.exists():
            raise NotADirectoryError("models dir is not found.")
        self.data_dir = self.root / "data"
        if not self.data_dir.exists():
            raise NotADirectoryError("data dir is not found.")
        self.arch_dir = self.root / "config" / "arch"
        if not self.arch_dir.exists():
            raise NotADirectoryError("arch dir is not found.")
        # model placeholder
        self.model = None
        self.version = None
        self.model_id = None
        self.embedding_results = {}

    def add_model(self, name, arch, version, model_id, d1=8, dn=10, w=20):
        arch = self.load_arch(arch)
        self.version = version
        self.model_id = model_id
        if self.model is None:
            self.model = Model(name, arch, d1=d1, dn=dn, w=w)
            self.model.load_model(version, model_id)

    def load_data(self, key, dataset_name):
        self.model.load_data(key, dataset_name)

    def load_arch(self, arch):
        arch_path = self.arch_dir / f"{arch}.json"
        return Architecture(read_json(str(arch_path)))

    def test(self, num_test_items=-1):
        self.model.test(num_test_items=num_test_items)

    def get_embedding(self, num_test_items=-1):
        for item in self.model.get_embedding(num_test_items=num_test_items):
            self.embedding_results[item.attrs["name"]] = item

    def tsne_embeddings(self, dim=2, perplexity=30):
        # combine embeddings
        all_embeddings = concat(
            [df.assign(pr=key) for key, df in self.embedding_results.items()], ignore_index=True
        )
        all_embeddings["uid"] = all_embeddings.apply(lambda x: f"{x.pr}_{x.unique_id}", axis=1)
        model = TSNE(
            n_components=dim,
            learning_rate="auto",
            init="pca",
            perplexity=perplexity,
            n_iter=100000,
            n_jobs=-1,
        )
        X_embedded = model.fit_transform(array(all_embeddings["embedding"].values.tolist()))
        for i in range(dim):
            all_embeddings[f"tsne_{i}"] = X_embedded[:, i]
        return all_embeddings

    def isomap_embeddings(self, dim=2, n_neighbors=5):
        # combine embeddings
        all_embeddings = concat(
            [df.assign(pr=key) for key, df in self.embedding_results.items()], ignore_index=True
        )
        all_embeddings["uid"] = all_embeddings.apply(lambda x: f"{x.pr}_{x.unique_id}", axis=1)
        model = Isomap(n_components=dim, n_neighbors=n_neighbors, n_jobs=-1)
        X_embedded = model.fit_transform(array(all_embeddings["embedding"].values.tolist()))
        for i in range(dim):
            all_embeddings[f"isomap_{i}"] = X_embedded[:, i]
        return all_embeddings

    def spectral_embeddings(self, dim=2, n_neighbors=5):
        # combine embeddings
        all_embeddings = concat(
            [df.assign(pr=key) for key, df in self.embedding_results.items()], ignore_index=True
        )
        all_embeddings["uid"] = all_embeddings.apply(lambda x: f"{x.pr}_{x.unique_id}", axis=1)
        model = SpectralEmbedding(n_components=dim, n_neighbors=n_neighbors, n_jobs=-1)
        X_embedded = model.fit_transform(array(all_embeddings["embedding"].values.tolist()))
        for i in range(dim):
            all_embeddings[f"spectral_{i}"] = X_embedded[:, i]
        return all_embeddings

    def plot_embedding(self, method="tsne", dim=2):
        t1 = datetime.now()
        filename = (
            self.models_dir
            / self.model.name
            / "versions"
            / self.version
            / f"tsne_results_{self.model_id}"
        )
        now = datetime.now().strftime("%Y%m%d%H%M")
        if not filename.exists():
            filename.mkdir()
        # calculate embeddings and tsne to dim dimensions
        if method == "tsne":
            for perplexity in [5, 10, 20, 30, 40, 50]:
                all_embeddings = self.tsne_embeddings(dim=dim, perplexity=perplexity)
                num_samples = len(unique(all_embeddings["pr"]))
                fig = px.scatter(
                    all_embeddings, x="tsne_0", y="tsne_1", color="pr", hover_data=["act_trg"]
                )
                html_filename = (
                    filename / f"{now}_tsne_dim_{dim}_p_{perplexity}_color_by_pr_{num_samples}.html"
                )
                plot(fig, filename=str(html_filename), auto_open=False)
                fig = px.scatter(
                    all_embeddings, x="tsne_0", y="tsne_1", color="act_trg", hover_data=["pr"]
                )
                html_filename = (
                    filename
                    / f"{now}_tsne_dim_{dim}_p_{perplexity}_color_by_act_{num_samples}.html"
                )
                plot(fig, filename=str(html_filename), auto_open=False)
        elif method == "isomap":
            for n_neighbors in [5, 10, 20, 50]:
                all_embeddings = self.isomap_embeddings(dim=dim, n_neighbors=n_neighbors)
                num_samples = len(unique(all_embeddings["pr"]))
                fig = px.scatter(
                    all_embeddings, x="isomap_0", y="isomap_1", color="pr", hover_data=["act_trg"]
                )
                html_filename = (
                    filename
                    / f"{now}_isomap_dim_{dim}_neighbors_{n_neighbors}_color_by_pr_{num_samples}.html"
                )
                plot(fig, filename=str(html_filename), auto_open=False)
                fig = px.scatter(
                    all_embeddings, x="isomap_0", y="isomap_1", color="act_trg", hover_data=["pr"]
                )
                html_filename = (
                    filename
                    / f"{now}_isomap_dim_{dim}_neighbors_{n_neighbors}_color_by_act_{num_samples}.html"
                )
                plot(fig, filename=str(html_filename), auto_open=False)
        elif method == "spectral":
            for n_neighbors in [None, 5, 10, 20, 50]:
                all_embeddings = self.spectral_embeddings(dim=dim, n_neighbors=n_neighbors)
                num_samples = len(unique(all_embeddings["pr"]))
                fig = px.scatter(
                    all_embeddings, x="spectral_0", y="isomap_1", color="pr", hover_data=["act_trg"]
                )
                html_filename = (
                    filename
                    / f"{now}_spectral_dim_{dim}_neighbors_{n_neighbors}_color_by_pr_{num_samples}.html"
                )
                plot(fig, filename=str(html_filename), auto_open=False)
                fig = px.scatter(
                    all_embeddings, x="spectral_0", y="isomap_1", color="act_trg", hover_data=["pr"]
                )
                html_filename = (
                    filename
                    / f"{now}_spectral_dim_{dim}_neighbors_{n_neighbors}_color_by_act_{num_samples}.html"
                )
                plot(fig, filename=str(html_filename), auto_open=False)
        print(datetime.now() - t1)
        # python ./SeqEN2/sessions/test_session.py -n dummy -mv 202201222143_AAECSS_arch7 -mid 0 -dcl kegg_ndx_ACTp_100 -a arch7 -teb 100 -ge -tsne 2


def main(args):
    # session
    test_session = TestSession()
    test_session.add_model(
        args["Model Name"],
        args["Arch"],
        args["Model Version"],
        args["Model ID"],
        d1=args["D1"],
        dn=args["Dn"],
        w=args["W"],
    )
    # load datafiles
    test_session.load_data("cl", args["Dataset_cl"])
    if args["Dataset_ss"] != "":
        test_session.load_data("ss", args["Dataset_ss"])
    if args["Dataset_clss"] != "":
        test_session.load_data("clss", args["Dataset_clss"])
    # tests
    # embeddings
    if args["Get Embedding"]:
        test_session.get_embedding(num_test_items=args["Test Batch"])
        if args["tSNE dim"]:
            test_session.plot_embedding(method="tsne", dim=args["tSNE dim"])
        if args["Isomap dim"]:
            test_session.plot_embedding(method="isomap", dim=args["Isomap dim"])
        if args["Spectral dim"]:
            test_session.plot_embedding(method="spectral", dim=args["Spectral dim"])


if __name__ == "__main__":
    # parse arguments
    parser = TestSessionArgParser()
    parsed_args = parser.parsed()
    main(parsed_args)
