ss_pattern_dict = {
    "(ACT-bbabba)": (["bbabba"], ["ACT"]),
    "(ACT-b(ACT-babbab)babba)": (["b-babba", "b-babba"], ["ACT", "ACT"]),
    "(ACT-b(UNK-babbab)babba)": (["b-babba", "b-babba"], ["UNK", "ACT"]),
    "(ACT-babbab)": (["babbab"], ["ACT"]),
    "(ACT-babbab)(ALS-babbab)": (["babbab", "bbabbabb"], ["ACT", "ALS"]),
    "(ACT-babbab)(ALS-babbab)(ACT-babbab)(ALS-babbab)": (
        ["babbab", "bbabbabb", "babbab", "bbabbabb"],
        ["ACT", "ALS", "ACT", "ALS"],
    ),
    "(ACT-babbab)(ALS-babbab)(ACT5-babbab)(ALS-babbab)": (
        ["babbab", "bbabbabb", "babbab", "bbabbabb"],
        ["ACT", "ALS", "ACT5", "ALS"],
    ),
    "(ACT-babbab)(UNK-babbab)": (["babbab", "babbab"], ["UNK", "ACT"]),
    "(ACT-bbabba)(ACT-babbab)(ACT-bbabba)(ACT-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["ACT", "ACT", "ACT", "ACT"],
    ),
    "(ACT-bbabba)(ACT-bbabba)(UNK-bbabba)(UNK-babbab)": (
        ["bbabba", "bbabba", "bbabba", "babbab"],
        ["ACT", "ACT", "UNK", "UNK"],
    ),
    "(ACT-bbabba)(ACT-babbab)(UNK-bbabba)(ACT-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["ACT", "ACT", "UNK", "ACT"],
    ),
    "(ACT-bbabba)(ACT-babbab)(UNK-bbabba)(UNK-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["ACT", "ACT", "UNK", "UNK"],
    ),
    "(ACT-bbabba)(ACT-bbabba)": (["bbabba", "bbabba"], ["ACT", "ACT"]),
    "(ACT-bbabba)(ACT6-babbab)(UNK-bbabba)(UNK-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["ACT", "ACT6", "UNK", "UNK"],
    ),
    "(ACT4-babbab)": (["babbab"], ["ACT4"]),
    "(ACT5-babbab)(ALS-babbab)(ACT5-babbab)(ALS-babbab)": (
        ["babbab", "bbabbabb", "babbab", "bbabbabb"],
        ["ACT5", "ALS", "ACT5", "ALS"],
    ),
    "(ACT5-babbab)(ALS-babbab)b": (["babbab", "bbabbabb"], ["ACT5", "ALS"]),
    "(ACT6-babbab)": (["babbab"], ["ACT6"]),
    "(ACT6-babbab)(ACT-babbab)": (["babbab", "babbab"], ["ACT6", "ACT"]),
    "(ACT6-babbab)(UNK-babbab)": (["babbab", "babbab"], ["ACT6", "UNK"]),
    "(ACT7-b(ACT-babba)b-babba)": (["b-babba", "b-babba"], ["ACT", "ACT7"]),
    "(ACT7-b(ACT-babba)b-babba)..(ACT7-b(ACT-babba)b-babba)": (
        ["b-babba", "b-babba", "b-babba", "b-babba"],
        ["ACT", "ACT7", "ACT", "ACT7"],
    ),
    "(ACT7-b(UNK-babba)b-babba)": (["b-babba", "b-babba"], ["UNK", "ACT7"]),
    "(ACT7-b(UNK-babba)b-babba)..(ACT7-b(UNK-babba)b-babba)": (
        ["b-babba", "b-babba", "b-babba", "b-babba"],
        ["UNK", "ACT7", "UNK", "ACT7"],
    ),
    "(ACT8-b-babbab-babbab)": (["babbab", "babbab"], ["ACT8", "ACT8"]),
    "(ALS-babbab)": (["bbabbabb"], ["ALS"]),
    "(NikR-babbab)": (["babbab"], ["NikR"]),
    "(UNK-b(ACT-babbab)babba)": (["b-babba", "b-babba"], ["ACT", "UNK"]),
    "(UNK-bbabba)(ACT-babbab)(UNK-bbabba)(ACT-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["UNK", "ACT", "UNK", "ACT"],
    ),
    "(UNK-bbabba)(ACT-babbab)(UNK-bbabba)(UNK-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["UNK", "ACT", "UNK", "UNK"],
    ),
    "(UNK-bbabba)(ACT-babbab)(UNK-bbabba)(UNK-bbabba)": (
        ["bbabba", "babbab", "bbabba", "bbabba"],
        ["UNK", "ACT", "UNK", "UNK"],
    ),
    "(UNK-bbabba)(ACT6-babbab)(UNK-bbabba)(ACT-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["UNK", "ACT6", "UNK", "ACT"],
    ),
    "(UNK-bbabba)(ACT6-babbab)(UNK-bbabba)(UNK-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["UNK", "ACT6", "UNK", "UNK"],
    ),
    "(UNK-bbabba)(UNK-babbab)(UNK-bbabba)(ACT-babbab)": (
        ["bbabba", "babbab", "bbabba", "babbab"],
        ["UNK", "UNK", "UNK", "ACT"],
    ),
    "(a-DUF-babbab)": (["babbab"], ["DUF"]),
    "(a-Thr-babbab-a)": (["babbab"], ["Thr"]),
    "(a-Thr-babbab-a)(a-Thr-babbab-a)": (["babbab", "babbab"], ["Thr", "Thr"]),
    "(b-ACT5-abbab-b)": (["babbab"], ["ACT5"]),
    "(b-NIL-abbab)": (["babbab"], ["NIL"]),
    "a(ACT7-b(ACT3-babbab)babba)": (["b-babba", "b-babba"], ["ACT3", "ACT7"]),
}
