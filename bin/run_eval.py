from os import system

models_dict = {
    "LONG": [
        ("202203050123_AECSS_arch70", 854, "arch70"),
    ],
    "DARWIN": [
        ("202204182125_AECSS_arch78", 99, "arch78"),
    ],
    "ISAAC": [
        ("202204200252_AECSS_arch79", 50, "arch79"),
    ],
}
for key, models in models_dict.items():
    for v, n, a in models:
        system(
            f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n {key} -mv {v} -mid {n} -dclss single_act_clss -a {a} -teb -1 -tsne 2 -ge"
        )
        system(
            f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n {key} -mv {v} -mid {n} -dclss clss -a {a} -teb -1 -ge"
        )
        # system(
        #     f"python3 ~/SeqEncoder/SeqEN/SeqEN2/sessions/test_session.py -n {key} -mv {v} -mid {n} -dcl cl_test -a {a} -teb -1 -ge"
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
