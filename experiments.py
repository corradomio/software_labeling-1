import csv
import json
import random as rnd
import sys
import netx
import netx.netxutils as ntxu
import netx.netxfun as ntxf
from collections import Counter
from datetime import datetime
from math import sqrt
from typing import Tuple, Dict, List, Union

import loggingx as logging
import networkx as nx
import numpy as np
from networkx.algorithms import node_classification as nclass
from path import Path as path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger(object):
    def __init__(self, filename="/python.log"):
        self.stdout = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        self.stdout.flush()
        self.log.flush()
# end


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.get_logger("main")
rnd.seed(datetime.now())


def sq(x): return x*x

def nop(): pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class Statistics:

    def __init__(self):
        self.data = []
    # end

    def add(self, x):
        assert type(x) in [float, int, np.float64]
        self.data.append(x)
    # end

    @property
    def mean(self):
        return sum(self.data)/len(self.data)

    @property
    def sdev(self):
        m = self.mean
        return sqrt(sum(map(lambda x: sq(x-m), self.data))/len(self.data))
    # end

    @property
    def min(self):
        return min(self.data)

    @property
    def max(self):
        return max(self.data)

    def print(self):
        print(f"    mean:{self.mean:.3} +- {self.sdev:.3}")
    # end
# end


def tolabels(cdict):
    nv = 0
    for c in cdict:
        nv = max(nv, max(cdict[c]))
    actual = [None]*(nv+1)
    for c in cdict:
        for v in cdict[c]:
            actual[v] = c
    return actual
# end


class ExperimentMetrics:

    def __init__(self, cats, cdict):
        self.cats = cats
        self.y_true = tolabels(cdict)

        self.ig = 0
        self.mode = None
        self.algo = None

        # self.accuracy = Statistics()
        # self.precision = Statistics()
        # self.recall = Statistics()
        # self.f1 = Statistics()
        self.results = []

        # nc = len(cats)
        # self.cm = np.zeros((nc, nc), dtype=int)
    # end

    def set_info(self, ig, mode, algo):
        self.ig = ig
        self.mode = mode
        self.algo = algo
    # end

    def predict(self, itry, selnodes, y_pred):
        y_true = self.y_true
        labels = self.cats

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1_ = f1_score(y_true, y_pred, average='macro')

        data = [
            self.ig, self.mode, self.algo, selnodes.k, selnodes.over_n,
            itry,
            acc, pre, rec, f1_
        ]

        tot = 0
        for c in labels:
            n = len(selnodes.selected[c])
            data.append(n)
            tot += n
        data.append(tot)

        self.results.append(data)

        # cm = confusion_matrix(y_true, y_pred, labels=labels)
        # self.cm += cm
        pass
    # end
# end


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

class SelectNodes:

    def __init__(self, g,
                 cdict: Dict[str, List[int]],
                 mode: str,
                 k: Union[int, Tuple[int, int], float, Tuple[float, float]]):
        self.g = g
        self.cdict = cdict
        self.mode = mode
        if type(k) in [int, float]:
            k = (k, k)
        self.k = k[0]
        self.over_n = k[1]
        self.selected = None
    # end

    def select(self) -> Dict[str, List[int]]:
        mode = self.mode
        cselected: Dict[str, List[int]] = dict()

        if mode == "layers":
            top, bottom = self._select_top_bottom()
        else:
            top, bottom = None, None

        for c in self.cdict:
            nodes = self.cdict[c]
            k = self.k if self.k >= 1 else max(1, int(self.k * len(nodes)))
            over_n = self.over_n if self.over_n >= 1 else max(1, int(self.over_n * len(nodes)))

            if mode == "rand":
                selected = self._select_rand(nodes, k, over_n)
            elif mode == "maxdeg":
                selected = self._select_maxdeg(nodes, k, over_n)
            elif mode == "layers":
                if c == top:
                    selected = self._select_layers(nodes, k, over_n, False)
                elif c == bottom:
                    selected = self._select_layers(nodes, k, over_n, True)
                else:
                    selected = self._select_maxdeg(nodes, k, over_n)
            elif mode == "centrality":
                selected = self._select_centrality(nodes, k, over_n)
            else:
                raise Exception("Invalid mode " + mode)
            cselected[c] = sorted(selected)
        # end
        self.selected = cselected
        return self.selected
    # end

    def _select_rand(self, nodes, k, over_n):
        selected = rnd.sample(nodes, k=k)
        return selected

    def _select_maxdeg(self, nodes, k, over_n):
        g = self.g
        degrees = list(map(lambda n: (n, g.degree[n]), nodes))
        degrees.sort(reverse=True, key=lambda d: d[1])
        nodes = list(map(lambda d: d[0], degrees[0:over_n]))
        selected = rnd.sample(nodes, k=k)
        return selected

    def _select_centrality(self, nodes, k, over_n):
        g = self.g
        degree_centrality = nx.degree_centrality(g)
        centrality = list(map(lambda n: (n, degree_centrality[n]), nodes))
        centrality.sort(reverse=True, key=lambda d: d[1])
        nodes = list(map(lambda d: d[0], centrality[0:over_n]))
        selected = rnd.sample(nodes, k=k)
        return selected

    def _select_layers(self, nodes, k, over_n, bottom):
        g = self.g
        if bottom:
            nodes = list(filter(lambda n: len(g.succ[n]) == 0, nodes))
            degrees = list(map(lambda n: (n, len(g.pred[n])), nodes))
        else:
            nodes = list(filter(lambda n: len(g.pred[n]) == 0, nodes))
            degrees = list(map(lambda n: (n, len(g.succ[n])), nodes))
        degrees.sort(reverse=True, key=lambda d: d[1])
        nodes = list(map(lambda d: d[0], degrees[0:over_n]))
        selected = rnd.sample(nodes, k=k)
        return selected

    def _select_top_bottom(self):
        def catof(n):
            for c in self.cdict:
                if n in self.cdict[c]:
                    return c
        # top layer
        g = self.g
        if not g.is_directed():
            return None, None

        nodes = list(g.nodes)
        # top: indegree == 0
        tnodes = list(filter(lambda n: len(g.pred[n]) == 0, nodes))
        tcounts = Counter(map(lambda n: catof(n), tnodes))
        top = tcounts.most_common()[0][0]
        # bottom: outdegree == 0
        bnodes = list(filter(lambda n: len(g.succ[n]) == 0, nodes))
        bcounts = Counter(map(lambda n: catof(n), bnodes))
        bottom = bcounts.most_common()[0][0]

        if top == bottom:
            return None, None
        else:
            return top, bottom
# end


class Experiments:

    def __init__(self,
                 label: str,
                 g: nx.Graph,
                 cdict: Union[Dict[str, List[int]], List[List[int]]],
                 rho: float = 0.60):

        assert not g.is_directed()

        if type(cdict) == list:
            cdict = {f"c{i:02}": cdict[i] for i in range(len(cdict))}

        # experiments label
        self.label: str = label
        # simple directed graph
        self.orig_g: nx.Graph = g
        self.g: nx.Graph = None
        # simple graph
        self.rho = rho

        # vertex list
        self.vlist = [v for v in g]
        # category -> list of vertices in that category (the order CHANGES)
        self.cdict = cdict
        # list of categories {c1,...}
        self.cats = list(self.cdict.keys())
        # top layer
        self.top_layer = None
        # bottom layer
        self.bottom_layer = None

        self.k: float = 0
        self.mode: str = ''
        self.algo: str = ''

        self.metrics = ExperimentMetrics(self.cats, self.cdict)
        self.results = self.metrics.results

        self.ig = 0
        self.itry = 0
        self.predicted = None

        self.nv = 0
        pass
    # end

    def do_experiments(self, n_graphs=100, n_tries=10):

        self._add_results_header()

        if n_graphs <= 1:
            self.g = self.orig_g
            self.apply_protocols(n_tries)
            self._save_data("spl-data.json")
            # self._dump_results()
        else:
            for ig in range(n_graphs):
                self.ig = ig
                self.gen_syntetic_graph()
                self.apply_protocols(n_tries)
            # end
            self._save_data("data.json")
            # self._dump_results()
        # end
    # end

    def gen_syntetic_graph(self):
        cdict = self.cdict
        rho = self.rho
        ncats = len(cdict)
        g = self.orig_g

        # retrieve the in/out degree of  the original graph
        degs = netx.degree_of(g, mode='degree')

        # create a synthetic graph with the same degrees (connected, max 10 retries)
        synthplain = ntxf.configuration_model(degs, create_using=nx.Graph(), connected=True)

        # save the graph
        netx.write_vecsv(synthplain, f"experiments/synthetic-swap-r{rho}c{ncats}.csv")
        ntxu.analyze_communities(synthplain, cdict, show=True, label=f"synth-plain-swap-r{rho}c{ncats}")

        # force the community
        synthetic = netx.force_communities(synthplain, cdict, rho=rho, mode="swap", connected=True)

        # assign the SAME node 'category' property based on original graph
        netx.set_node_attribute(synthetic, attribute='category', communities=cdict)
        netx.write_vecsv(synthetic, f"experiments/synthetic-communities-swap-r{rho}c{ncats}.csv")

        # dump the communities statistics
        print("--- Synthetic (communities) ---")
        ntxu.analyze_communities(synthetic, cdict, show=True, label=f"synth-force-swap-r{rho}c{ncats}")

        # save the dict
        self.g = synthetic
        pass
    # end

    def apply_protocols(exp, n_tries=10):
        for algo in ['lgc', 'hf']:

            print(f"\n=== random k ({n_tries}) ===")
            for k in [1, 2, 3, 4, 5]:
                exp.experiment_loop(k, 'rand', algo=algo, n_tries=n_tries)
            print(f"\n=== random r*Nc ({n_tries}) ===")
            for r in [.01, .02, .03, .04, .05]:
                exp.experiment_loop(r, 'rand', algo=algo, n_tries=n_tries)

            try:
                print("\n=== layers (k over n) ===")
                n = 10
                for k in [(1, n), (2, n), (3, n), (4, n), (5, n)]:
                    exp.experiment_loop(k, 'layers', algo=algo, n_tries=n_tries)
                print("\n=== layers (r*Nc over s*Nc) ===")
                n = .1
                for k in [(.01, n), (.02, n), (.03, n), (.04, n), (.05, n)]:
                    exp.experiment_loop(k, 'layers', algo=algo, n_tries=n_tries)
            except:
                pass

            print("\n=== maxdeg (k over n) ===")
            n = 10
            for k in [(1, n), (2, n), (3, n), (4, n), (5, n)]:
                exp.experiment_loop(k, 'maxdeg', algo=algo, n_tries=n_tries)
            print("\n=== maxdeg (r*Nc over s*Nc) ===")
            n = .1
            for k in [(.01, n), (.02, n), (.03, n), (.04, n), (.05, n)]:
                exp.experiment_loop(k, 'maxdeg', algo=algo, n_tries=n_tries)

            print("\n=== centrality k ===")
            for k in [1, 2, 3, 4, 5]:
                exp.experiment_loop(k, 'centrality', algo=algo)
            print("\n=== centrality r*Nc ===")
            for r in [.01, .02, .03, .04, .05]:
                exp.experiment_loop(r, 'centrality', algo=algo)

            print("=== End ===")
        # end
        exp.save_results()
    # end

    def experiment_loop(self, k: Union[int, float, tuple], mode: str = '', algo: str = '', n_tries: int = 1):
        """

        The results will be saved in the file 'results.csv'

        :param k: n of nodes to select
        :param mode: how to select the nodes

                - rand          randomly
                - maxdeg        nodes with highest degree
                - layers        nodes with highest degree, based on the 'layer'
                                'top layer' contains nodes with input degree = 0
                                'bottom layer' contains nodes with output degree = 0
                - centrality

        :param algo: algorithm to use for label propagation

                - lgc: local global ..
                - ...

        :param n_tries: n of times the experiment is executes
        """
        self.k = k
        self.mode = mode
        self.algo = algo if len(algo) > 0 else 'lgc'

        # self._add_results_header()

        self._run_experiments(n_tries)

        # self._collect_results()
        # self._save_data("data.json")
        # self._dump_results()
    # end

    def _add_results_header(self):
        if len(self.results) > 0:
            return

        # add the FIRST data row, containing GLOBAL information
        # ig, mode, algorithm, itry, k, over_n,
        #   acc
        #   prc
        #   rcl
        #   f1
        info = [0, 'info', 'lgc-hf', 0, 0, 0,
                0, 0, 0, 0
        ]

        # |V[cat1]|, |V[cat2]|, ....
        total = 0
        for c in self.cdict:
            nc = len(self.cdict[c])
            total += nc
            info.append(nc)
        # |V|
        info.append(total)
    # end

    def _run_experiments(self, n_tries: int):
        # execute experiments
        self.metrics.set_info(self.ig, self.mode, self.algo)
        nop()
        for itry in range(n_tries):
            self.itry = itry
            self._experiment()
        pass
    # end

    def _experiment(self):
        itry = self.itry
        selnodes = SelectNodes(self.g, self.cdict, self.mode, self.k)
        assigned = selnodes.select()
        # print(assigned)
        self.nv = sum(len(assigned[c]) for c in assigned)

        # assign the categories
        self._assign_cats(assigned)
        # predict all categories
        predicted = self._predict()

        # metrics
        self.metrics.predict(itry, selnodes, predicted)
    # end

    def _assign_cats(self, assigned: Dict[str, List[int]]):
        # clear previous classification
        nv = 0
        for v in self.vlist:
            if 'label' in self.g.nodes[v]:
                nv += 1
                del self.g.nodes[v]['label']
                # assign the new classification
        nv = 0
        for c in assigned:
            for v in assigned[c]:
                nv += 1
                self.g.nodes[v]['label'] = c
        # save the assigned labels
        self.assigned = assigned
    # end

    def _predict(self):
        g = self.g
        # log.info("classify")
        # g = nx.to_undirected(self.g)
        # g = self.g.to_undirected()
        if self.algo == 'lgc':
            self.predicted = nclass.local_and_global_consistency(g)
        else:
            self.predicted = nclass.harmonic_function(g)
        return self.predicted
    # end

    def _compose_header(self) -> List[str]:
        header = [
            "ig",
            "mode", "algo", "k", "over_n",
            "i",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]

        # compose counts on labels
        for c in self.cats:
            header.append(f"#nodes[{c}]")
        header.append("#nodes")
        return header
    # end

    # def _dump_results(self):
    #     k = self.k
    #     nv = self.nv
    #     if type(k) in [tuple, list]:
    #         k, n = k
    #         if k >= 1:
    #             print(f"n: {k: 2} over {n: 2} ({nv: 3}) -> acc: {self.metrics.accuracy.mean:0.3f} +- {self.metrics.accuracy.sdev:0.3f}")
    #         else:
    #             print(f"r: {k:.2} over {n:.2} ({nv: 3}) -> acc: {self.metrics.accuracy.mean:0.3f} +- {self.metrics.accuracy.sdev:0.3f}")
    #     else:
    #         if k >= 1:
    #             print(f"n: {k: 2} ({nv: 3}) -> acc: {self.metrics.accuracy.mean:0.3f} +- {self.metrics.accuracy.sdev:0.3f}")
    #         else:
    #             print(f"r: {k:.2} ({nv: 2}) -> acc: {self.metrics.accuracy.mean:0.3f} +- {self.metrics.accuracy.sdev:0.3f}")
    #     # end
    #     print("== --- ==")
    # # end

    def _save_data(self, basename, parent="experiments"):
        homedir = path("{}/{}".format(parent, self.label))
        homedir.makedirs_p()

        if type(self.k) in [tuple, list]:
            k, n = self.k
            if k >= 1:
                fname = "{}/{}/{}-k{}-n{}-{}.json".format(parent, self.label, self.mode, k, n, self.algo, basename)
            else:
                fname = "{}/{}/{}-p{}-q{}-{}.json".format(parent, self.label, self.mode, int(k*100), int(n*100), self.algo, basename)
        else:
            if self.k >= 1:
                fname = "{}/{}/{}-k{}-{}.json".format(parent, self.label, self.mode, self.k, self.algo, basename)
            else:
                fname = "{}/{}/{}-p{}-{}.json".format(parent, self.label, self.mode, int(self.k*100), self.algo, basename)
        # end

        predicted = self._predicted()

        data = {
            "algo": self.algo,
            "cats": self.cats,
            "selected": self.assigned,
            "predicted": predicted,
        }

        if self.top_layer is not None:
            data.update({
                "top_layer": self.top_layer,
                "bottom_layer": self.bottom_layer
            })

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=2)
    # end

    def _predicted(self):
        dpred = {c: [] for c in self.cats}
        vlist = self.vlist
        predicted = self.predicted
        n = len(vlist)

        for i in range(n):
            v = vlist[i]
            c = predicted[i]
            dpred[c].append(v)
        # end
        return dpred
    # end

    def save_results(self, parent="experiments"):
        fname = "{}/{}/results.csv".format(parent, self.label)

        with open(fname, mode='w', newline='') as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='\'')
            header = self._compose_header()
            csvwriter.writerow(header)
            csvwriter.writerows(self.results)
        # end
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
