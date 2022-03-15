from numpy import vstack

from SeqEN2.sessions.test_session import TestSession

session = TestSession()

session.add_model("AECSS", "arch70", "202203050123_AECSS_arch70", 45, d1=8, dn=10, w=20)

# loading clss data
session.load_data("clss", "single_act_clss_test")

session.load_embeddings()

sample_embedding, aff50 = session.tsne_embeddings_2(return_embeddings=True)

# loading cl data
session.load_data("clss", "kegg_ndx_ACTp_test")

session.load_embeddings()
# filter for act
session.all_embeddings = session.all_embeddings[session.all_embeddings["pred_class"] > 0.8]

rest_init = sample_embedding.prepare_partial(
    session.get_embeddings_from_df(), k=1, perplexity=1 / 3
)
init_full = vstack((sample_embedding, rest_init))

session.tsne_embeddings_from_init(init_full, aff50, len(sample_embedding))
