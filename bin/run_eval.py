from os import system

models = {
    "AECSS": [
        # ("202203050123_AECSS_arch70", 199, "arch70"),
        # ("202203050123_AECSS_arch70", 299, "arch70"),
        # ("202203050123_AECSS_arch70", 399, "arch70"),
        # ("202203050123_AECSS_arch70", 499, "arch70"),
        # ("202203050123_AECSS_arch70", 599, "arch70"),
        # ("202203050123_AECSS_arch70", 699, "arch70"),
        # ("202203050123_AECSS_arch70", 750, "arch70"),
        # ("202203050123_AECSS_arch70", 799, "arch70"),
        ("202203050123_AECSS_arch70", 854, "arch70"),
        # ("202203050123_AECSS_arch70", 999, "arch70"),
        # ("202203050123_AECSS_arch70", 499, "arch70"),
        # ("202203050123_AECSS_arch70", 599, "arch70"),
        # ("202203050123_AECSS_arch70", 699, "arch70"),
        # ("202203050123_AECSS_arch70", 750, "arch70"),
    ]
}
for v, n, a in models["AECSS"]:
    system(
        f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dclss clss -a {a} -teb -1 -ge"
    )
    system(
        f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dclss single_act_clss -a {a} -teb -1 -tsne 2 -ge"
    )
    system(
        f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dcl cl_test -a {a} -teb -1 -ge"
    )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dcl cl_train -a {a} -teb -1 -ge"
    # )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dss ss_test -a {a} -teb -1 -ge"
    # )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dss ss_train -a {a} -teb -1 -ge"
    # )
