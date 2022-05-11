#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from datetime import datetime
from glob import glob
from os.path import dirname
from pathlib import Path

import openTSNE
import plotly.express as px
from numpy import array, sqrt, unique
from pandas import concat, read_pickle
from plotly.offline import plot
from sklearn.manifold import TSNE
from tqdm import tqdm

from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import read_json
from SeqEN2.model.model import Model
from SeqEN2.utils.custom_arg_parser import TestSessionArgParser


def now():
    print(datetime.now().strftime("%Y%m%d%H%M%S"))


EXCEPTIONS = ["AF-A0A1D8PLB7-F1-model_v1_1", "AF-O60086-F1-model_v1_1"]


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
        self.all_embeddings = None
        # run dir
        self.result_dir = None
        # attrs
        self.smooth_embed = False

    def add_model(self, name, arch, version, model_id):
        arch = self.load_arch(arch)
        self.version = version
        self.model_id = model_id
        if self.model is None:
            self.model = Model(name, arch)
            self.model.load_model(version, model_id)
        self.result_dir = self.models_dir / self.model.name / "results" / self.version
        if not self.result_dir.exists():
            self.result_dir.mkdir()

    def load_data(self, key, dataset_name):
        self.model.eval_only = True
        self.model.load_eval_data(key, dataset_name)

    def load_arch(self, arch):
        arch_path = self.arch_dir / f"{arch}.json"
        return Architecture(read_json(str(arch_path)))

    def test(self, num_test_items=-1):
        self.model.test(num_test_items=num_test_items)

    def get_embedding(self, num_test_items=-1, test_items=None):
        print("embedding proteins ....")
        now()
        # embeddings dir
        embeddings_dir = (
            self.result_dir / f"embeddings_only_{self.model_id}_{self.model.eval_data_loader_name}"
        )
        if not embeddings_dir.exists():
            embeddings_dir.mkdir()
        # getting embeddings
        self.embedding_results = {}
        for item in tqdm(
            self.model.get_embedding(num_test_items=num_test_items, test_items=test_items)
        ):
            if item.attrs["name"] in EXCEPTIONS:
                continue
            if len(self.embedding_results) < 1000:
                self.embedding_results[item.attrs["name"]] = item
            datafile = embeddings_dir / f"{item.attrs['name']}.pkl.bz2"
            item.to_pickle(datafile)
        now()

    def tsne_embeddings(self, dim=2):
        # combine embeddings
        print("tsne Embeddings ...")
        now()
        self.all_embeddings = concat(
            [df.assign(pr=key) for key, df in self.embedding_results.items()], ignore_index=True
        )
        self.all_embeddings["uid"] = self.all_embeddings.apply(
            lambda x: f"{x.pr}_{x.unique_id}", axis=1
        )
        perplexity = sqrt(len(self.all_embeddings["uid"]))
        model = TSNE(
            n_components=dim,
            learning_rate="auto",
            init="pca",
            perplexity=perplexity,
            n_iter=10000,
            n_jobs=-1,
        )
        x_embedded = model.fit_transform(self.get_embeddings_from_df())
        for i in range(dim):
            self.all_embeddings[f"tsne_{i}"] = x_embedded[:, i]

    def get_embeddings_from_df(self):
        return array(self.all_embeddings["embedding"].values.tolist())

    def tsne_embeddings_from_init(self, init, split):
        aff50 = openTSNE.affinity.PerplexityBasedNN(
            init,
            perplexity=100,
            n_jobs=32,
            random_state=0,
            verbose=True,
        )
        embedding = openTSNE.TSNEEmbedding(
            init,
            aff50,
            n_jobs=32,
            verbose=True,
            random_state=42,
        )
        embedding1 = embedding.optimize(n_iter=500, exaggeration=12, momentum=0.5)
        embedding2 = embedding1.optimize(n_iter=250, exaggeration=12, momentum=0.8)
        embedding3 = embedding2.optimize(n_iter=250, exaggeration=12, momentum=0.8)
        embedding4 = embedding3.optimize(n_iter=250, exaggeration=12, momentum=0.8)
        embedding = embedding4[split:]

        for i in range(2):
            self.all_embeddings[f"tsne_{i}"] = embedding[:, i].tolist()

    def tsne_embeddings_2(self, dim=2, return_embeddings=False):
        # combine embeddings
        print("tsne Embeddings ...")
        now()
        if self.all_embeddings is None and len(self.embedding_results) > 0:
            self.all_embeddings = concat(
                [df.assign(pr=key) for key, df in self.embedding_results.items()], ignore_index=True
            )
            self.all_embeddings["uid"] = self.all_embeddings.apply(
                lambda x: f"{x.pr}_{x.unique_id}", axis=1
            )
        exaggeration = 12
        data = self.get_embeddings_from_df()
        aff50 = openTSNE.affinity.PerplexityBasedNN(
            data,
            perplexity=100,
            n_jobs=32,
            random_state=0,
            verbose=True,
        )
        init = openTSNE.initialization.pca(data, random_state=0)
        embedding_standard = openTSNE.TSNE(
            exaggeration=exaggeration,
            n_jobs=32,
            verbose=True,
        ).fit(affinities=aff50, initialization=init)
        for i in range(dim):
            self.all_embeddings[f"tsne_{i}"] = embedding_standard[:, i].tolist()

        if return_embeddings:
            return embedding_standard
        else:
            return None

    def load_embeddings(self, save_aggregate=False):
        embeddings_dir = (
            self.result_dir / f"embeddings_only_{self.model_id}_{self.model.eval_data_loader_name}"
        )
        files = glob(f"{embeddings_dir}/*.pkl.bz2")
        self.all_embeddings = concat(
            [read_pickle(fp).assign(pr=fp.split(".")[0]) for fp in tqdm(files)], ignore_index=True
        )
        self.all_embeddings["uid"] = self.all_embeddings.apply(
            lambda x: f"{x.pr}_{x.unique_id}", axis=1
        )
        if save_aggregate:
            # embeddings dir
            embeddings_dir = (
                self.result_dir / f"tsne_{self.model_id}_{self.model.eval_data_loader_name}"
            )
            datafile = embeddings_dir / f"all_tsne_dim_2.pkl.bz2"
            self.all_embeddings.to_pickle(datafile)

    def plot_embedding_2d(self, auto_open=False, pr_ids=None, text=""):
        if pr_ids is None:
            pr_ids = []
        # embeddings dir
        plots_dir = (
            self.result_dir / f"embeddings_plots_{self.model_id}_{self.model.eval_data_loader_name}"
        )
        if not plots_dir.exists():
            plots_dir.mkdir()
        # embeddings dir
        embeddings_dir = (
            self.result_dir / f"tsne_{self.model_id}_{self.model.eval_data_loader_name}"
        )
        datafile = embeddings_dir / f"tsne_dim_2.pkl.bz2"
        if not embeddings_dir.exists():
            embeddings_dir.mkdir()

        if not datafile.exists():
            if self.all_embeddings is None:
                self.tsne_embeddings_2(dim=2)
            self.all_embeddings["size"] = self.all_embeddings["pred_class"] + self.MIN_SPOT_SIZE
            self.all_embeddings.to_pickle(datafile)
        else:
            self.all_embeddings = read_pickle(datafile)
            if len(pr_ids) > 0:
                pattern = "|".join(pr_ids)
                mask = self.all_embeddings["pr"].str.contains(pattern, case=False, na=False)
                self.all_embeddings = self.all_embeddings[mask]
        print("tsne Done.")
        text = f"_{text}" if text != "" else ""
        now()
        # calculate embeddings and tsne to dim dimensions
        num_samples = len(unique(self.all_embeddings["pr"]))
        fig = px.scatter(
            self.all_embeddings,
            x="tsne_0",
            y="tsne_1",
            color="pr",
            hover_data=[
                "w_seq",
                "w_cons_seq",
                "w_trg_class",
                "pred_class",
                "w_trg_ss",
                "w_cons_ss",
                "pr",
            ],
            size="size",
        )
        html_filename = plots_dir / f"tsne_dim_2_color_by_pr_{num_samples}{text}.html"
        plot(fig, filename=str(html_filename), auto_open=auto_open)
        ####
        fig = px.scatter(
            self.all_embeddings,
            x="tsne_0",
            y="tsne_1",
            color="w_trg_class",
            hover_data=[
                "w_seq",
                "w_cons_seq",
                "w_trg_class",
                "pred_class",
                "w_trg_ss",
                "w_cons_ss",
                "pr",
            ],
            size="size",
        )
        html_filename = plots_dir / f"tsne_dim_2_color_by_act_{num_samples}{text}.html"
        plot(fig, filename=str(html_filename), auto_open=auto_open)
        ####
        fig = px.line(
            self.all_embeddings,
            x="tsne_0",
            y="tsne_1",
            color="pr",
            hover_data=[
                "w_seq",
                "w_cons_seq",
                "w_trg_class",
                "pred_class",
                "w_trg_ss",
                "w_cons_ss",
                "pr",
            ],
            markers=True,
            render_mode="svg",
            line_shape="spline",
        )
        html_filename = plots_dir / f"tsne_dim_2_color_by_pr_lines_{num_samples}{text}.html"
        plot(fig, filename=str(html_filename), auto_open=auto_open)
        #####
        fig = px.scatter(
            self.all_embeddings,
            x="tsne_0",
            y="tsne_1",
            color="w_trg_class",
            hover_data=[
                "w_seq",
                "w_cons_seq",
                "w_trg_class",
                "pred_class",
                "w_trg_ss",
                "w_cons_ss",
                "pr",
            ],
        )
        fig.update_traces(
            marker=dict(size=1, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        html_filename = plots_dir / f"tsne_dim_2_color_by_act_small_{num_samples}{text}.html"
        plot(fig, filename=str(html_filename), auto_open=auto_open)
        # python ./SeqEN2/sessions/test_session.py -n dummy -mv 202201222143_AAECSS_arch7 -mid 0 -dcl kegg_ndx_ACTp_100 -a arch7 -teb 100 -ge -tsne 2
        # python3 ../../../SeqEN2/sessions/test_session.py -n AECSS -mv 202203042153_AECSS_arch66 -mid 24 -dclss single_act_clss_test -a arch66
        # -teb -1 -ge


def main(args):
    # session
    test_session = TestSession()
    test_session.add_model(
        args["Model Name"], args["Arch"], args["Model Version"], args["Model ID"]
    )
    test_session.model.embed_only = args["Embed Only"]
    test_session.smooth_embed = args["Smooth Embed"]
    # load datafiles
    if args["Dataset_cl"] != "":
        test_session.load_data("cl", args["Dataset_cl"])
    elif args["Dataset_ss"] != "":
        test_session.load_data("ss", args["Dataset_ss"])
    elif args["Dataset_clss"] != "":
        test_session.load_data("clss", args["Dataset_clss"])
    # tests
    # embeddings
    if args["Get Embedding"]:
        test_session.get_embedding(num_test_items=args["Test Batch"])
    if args["tSNE dim"]:
        if args["tSNE dim"] == 2:
            # pr_ids = args["Pr IDs"].split(",") if args["Pr IDs"] != "" else None
            text = args["Text"]
            test_session.plot_embedding_2d(text=text)


if __name__ == "__main__":
    # parse arguments
    parser = TestSessionArgParser()
    parsed_args = parser.parsed()
    main(parsed_args)
