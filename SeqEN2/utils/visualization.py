from ete3 import PhyloTree, Tree, TreeStyle


def get_example_tree(alg, tree):

    # Performs a tree reconciliation analysis
    genetree = PhyloTree(tree)
    genetree.link_to_alignment(alg)
    return genetree, TreeStyle()


if __name__ == "__main__":
    # Visualize the reconciled tree
    with open("../../../data/single_act_clss_align.fasta", "r") as file:
        alg = file.read()
    with open("../../../data/single_ACT_structure_tree.txt", "r") as file:
        tree = file.read()
    t, ts = get_example_tree(alg, tree)
    t.show(tree_style=ts)
    # recon_tree.render("phylotree.png", w=750)
