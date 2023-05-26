from experiments import *
from netx import netxutils as ntxu


def try_params(rho, force=False):
    # retrieve the map 'namespace->category'
    catmap = ntxu.load_community_map("data/jdeps-categories-c7.csv")
    ncats = len(set(catmap.values()))
    label = f"jdeps.synthetic.swap.r{rho}c{ncats}"
    expdir = path(f"experiments/{label}")
    if expdir.exists() and not force:
        return

    expdir.makedirs_p()
    sys.stdout = Logger(f"experiments/{label}/results.log")

    # 1) read the graph
    g: nx.Graph = netx.read_odem("data/jdeps-2.8.0-connected.odem", create_using=nx.Graph())

    assert not g.is_directed()
    assert ntxf.is_connected(g)

    # retrieve the classification based on the 'namespace' property
    cdict: Dict[str, List[int]] = netx.communities_on_attribute(g, 'namespace')

    # assign the node 'category' property based on 'namespace' and map 'namespace->category'
    netx.set_node_attribute(g, attribute='category', communities=cdict, map=catmap)

    # retrieve the classification based on the 'category' property
    cdict: Dict[str, List[int]] = netx.communities_on_attribute(g, 'category')
    ntxu.write_vertex_category(f'experiments/synthetic-swap-r{rho}c{ncats}-categories.dat', cdict)

    # dump the communities statistics
    print("--- Original ---")
    ntxu.analyze_communities(g, cdict, show=True, label="original")

    # do the experiments
    exp = Experiments(label, g, cdict, rho=rho)

    exp.do_experiments(n_graphs=100, n_tries=10)
# end


def main():
    # rhos = [0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # rhos = [0.80]
    # rhos = [0.85]
    # rhos = [0.65]
    # rhos = [0.75]
    rhos = [0.65, 0.7, 0.75, 0.8, 0.85]

    # execute sequentially
    for rho in rhos:
        try_params(rho, force=True)

    # execute in parallel
    # Parallel(n_jobs=4)(delayed(try_params)(rho) for rho in rhos)
    pass
# end


if __name__ == "__main__":
    logging.Logger.configure_level(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger('PIL').setLevel(logging.WARN)
    main()

