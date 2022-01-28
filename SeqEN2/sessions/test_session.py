#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from datetime import datetime
from os.path import dirname
from pathlib import Path

import plotly.express as px
from numpy import array, sqrt, unique
from pandas import concat
from plotly.offline import plot
from sklearn.manifold import TSNE, Isomap

from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import read_json
from SeqEN2.model.model import Model
from SeqEN2.utils.custom_arg_parser import TestSessionArgParser


class TestSession:

    root = Path(dirname(__file__)).parent.parent
    MIN_SPOT_SIZE = 0.05

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

    def get_embedding(self, num_test_items=-1, test_items=None):
        for item in self.model.get_embedding(num_test_items=num_test_items, test_items=test_items):
            self.embedding_results[item.attrs["name"]] = item

    def tsne_embeddings(self, dim=2):
        # combine embeddings
        all_embeddings = concat(
            [df.assign(pr=key) for key, df in self.embedding_results.items()], ignore_index=True
        )
        all_embeddings["uid"] = all_embeddings.apply(lambda x: f"{x.pr}_{x.unique_id}", axis=1)
        perplexity = sqrt(len(all_embeddings["uid"]))
        model = TSNE(
            n_components=dim,
            learning_rate="auto",
            init="pca",
            perplexity=perplexity,
            n_iter=10000,
            n_jobs=-1,
        )
        X_embedded = model.fit_transform(array(all_embeddings["embedding"].values.tolist()))
        for i in range(dim):
            all_embeddings[f"tsne_{i}"] = X_embedded[:, i]
        return all_embeddings

    def isomap_embeddings(self, dim=2):
        # combine embeddings
        all_embeddings = concat(
            [df.assign(pr=key) for key, df in self.embedding_results.items()], ignore_index=True
        )
        all_embeddings["uid"] = all_embeddings.apply(lambda x: f"{x.pr}_{x.unique_id}", axis=1)
        n_neighbors = int(sqrt(len(all_embeddings["uid"])))
        model = Isomap(n_components=dim, n_neighbors=n_neighbors, n_jobs=-1)
        X_embedded = model.fit_transform(array(all_embeddings["embedding"].values.tolist()))
        for i in range(dim):
            all_embeddings[f"isomap_{i}"] = X_embedded[:, i]
        return all_embeddings

    def plot_embedding_2d(self, method="tsne"):
        t1 = datetime.now()
        filename = (
            self.models_dir
            / self.model.name
            / "versions"
            / self.version
            / f"embeddings_results_{self.model_id}"
        )
        now = datetime.now().strftime("%Y%m%d%H%M")
        if not filename.exists():
            filename.mkdir()
        # calculate embeddings and tsne to dim dimensions
        if method == "tsne":
            all_embeddings = self.tsne_embeddings(dim=2)
            all_embeddings["size"] = all_embeddings["act_pred"] + MIN_SPOT_SIZE
            num_samples = len(unique(all_embeddings["pr"]))
            fig = px.scatter(
                all_embeddings,
                x="tsne_0",
                y="tsne_1",
                color="pr",
                hover_data=["act_trg", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_tsne_dim_{2}_color_by_pr_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            fig = px.scatter(
                all_embeddings,
                x="tsne_0",
                y="tsne_1",
                color="act_trg",
                hover_data=["pr", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_tsne_dim_{2}_color_by_act_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            datafile = filename / f"{now}_tsne_dim_{2}.pkl.bz2"
            all_embeddings.to_pickle(datafile)
        elif method == "isomap":
            all_embeddings = self.isomap_embeddings(dim=2)
            all_embeddings["size"] = all_embeddings["act_pred"] + self.MIN_SPOT_SIZE
            num_samples = len(unique(all_embeddings["pr"]))
            fig = px.scatter(
                all_embeddings,
                x="isomap_0",
                y="isomap_1",
                color="pr",
                hover_data=["act_trg", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_isomap_dim_{2}_color_by_pr_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            fig = px.scatter(
                all_embeddings,
                x="isomap_0",
                y="isomap_1",
                color="act_trg",
                hover_data=["pr", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_isomap_dim_{2}_color_by_act_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            datafile = filename / f"{now}_isomap_dim_{2}.pkl.bz2"
            all_embeddings.to_pickle(datafile)
        print(datetime.now() - t1)

    def plot_embedding_3d(self, method="tsne"):
        t1 = datetime.now()
        filename = (
            self.models_dir
            / self.model.name
            / "versions"
            / self.version
            / f"embeddings_results_{self.model_id}"
        )
        now = datetime.now().strftime("%Y%m%d%H%M")
        if not filename.exists():
            filename.mkdir()
        # calculate embeddings and tsne to dim dimensions
        if method == "tsne":
            all_embeddings = self.tsne_embeddings(dim=3)
            all_embeddings["size"] = all_embeddings["act_pred"] + self.MIN_SPOT_SIZE
            num_samples = len(unique(all_embeddings["pr"]))
            fig = px.scatter_3d(
                all_embeddings,
                x="tsne_0",
                y="tsne_1",
                z="tsne_2",
                color="pr",
                hover_data=["act_trg", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_tsne_dim_{3}_color_by_pr_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            fig = px.scatter_3d(
                all_embeddings,
                x="tsne_0",
                y="tsne_1",
                z="tsne_2",
                color="act_trg",
                hover_data=["pr", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_tsne_dim_{3}_color_by_act_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            datafile = filename / f"{now}_tsne_dim_{3}.pkl.bz2"
            all_embeddings.to_pickle(datafile)
            print(datetime.now() - t1)
            t1 = datetime.now()

        elif method == "isomap":
            all_embeddings = self.isomap_embeddings(dim=3)
            all_embeddings["size"] = all_embeddings["act_pred"] + MIN_SPOT_SIZE
            num_samples = len(unique(all_embeddings["pr"]))
            fig = px.scatter_3d(
                all_embeddings,
                x="isomap_0",
                y="isomap_1",
                z="isomap_2",
                color="pr",
                hover_data=["act_trg", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_isomap_dim_{3}_color_by_pr_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            fig = px.scatter_3d(
                all_embeddings,
                x="isomap_0",
                y="isomap_1",
                z="isomap_2",
                color="act_trg",
                hover_data=["pr", "act_pred", "slices"],
                size="size",
            )
            html_filename = filename / f"{now}_isomap_dim_{3}_color_by_act_{num_samples}.html"
            plot(fig, filename=str(html_filename), auto_open=False)
            datafile = filename / f"{now}_isomap_dim_{3}.pkl.bz2"
            all_embeddings.to_pickle(datafile)
            print(datetime.now() - t1)
            t1 = datetime.now()
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
            if args["tSNE dim"] == 2:
                test_session.plot_embedding_2d(method="tsne")
            elif args["tSNE dim"] == 3:
                test_session.plot_embedding_3d(method="tsne")
        if args["Isomap dim"]:
            if args["Isomap dim"] == 2:
                test_session.plot_embedding_2d(method="isomap")
            elif args["Isomap dim"] == 3:
                test_session.plot_embedding_3d(method="isomap")


if __name__ == "__main__":
    # parse arguments
    parser = TestSessionArgParser()
    parsed_args = parser.parsed()
    main(parsed_args)
