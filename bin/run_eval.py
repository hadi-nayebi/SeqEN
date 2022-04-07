from os import system

models = {
    "AECSS": [
        # ("202203042153_AECSS_arch66", 24, "arch66"),
        # ("202203050040_AECSS_arch69", 24, "arch69"),
        ("202203050123_AECSS_arch70", 544, "arch70"),
        # ("202203042153_AECSS_arch66_dcl", 29, "arch66"),
    ]
}
for v, n, a in models["AECSS"]:
    system(
        f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dclss clss -a {a} -teb -1 -ge"
    )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dclss single_act_clss -a {a} -teb -1 -tsne 2 -ge"
    # )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dcl cl_test -a {a} -teb -1 -ge"
    # )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dcl cl_train -a {a} -teb -1 -ge"
    # )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dss ss_test -a {a} -teb -1 -ge"
    # )
    # system(
    #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n LONG -mv {v} -mid {n} -dss ss_train -a {a} -teb -1 -ge"
    # )
