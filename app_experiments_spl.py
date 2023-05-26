import netx
from experiments import *
from netx import netxutils as ntxu


def try_params(rho, force=False):
    # retrieve the map 'namespace->category'
    label = f"spl"
    expdir = path(f"experiments/{label}")
    if expdir.exists() and not force:
        return

    expdir.makedirs_p()
    sys.stdout = Logger(f"experiments/{label}/results.log")

    # 1) read the graph
    g: nx.Graph = netx.read_vecsv("data/4c00c2ca-component-graph-1-r00-edges.csv", create_using=nx.Graph())

    # retrieve the classification based on the 'category' property
    cdict: Dict[str, List[int]] = netx.communities_on_attribute(g, 'category')
    ntxu.write_vertex_category(f'experiments/spl/spl-categories.dat', cdict)

    # dump the communities statistics
    print("--- Original ---")
    ntxu.analyze_communities(g, cdict, show=True, label="spl-original")

    # do the experiments
    exp = Experiments(label, g, cdict, rho=rho)

    exp.do_experiments(n_graphs=1, n_tries=10)
# end


def main():
    rhos = [1.0]

    for rho in rhos:
        try_params(rho, force=True)

    # Parallel(n_jobs=4)(delayed(try_params)(rho) for rho in rhos)
    pass
# end


if __name__ == "__main__":
    logging.Logger.configure_level(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger('PIL').setLevel(logging.WARN)
    main()

