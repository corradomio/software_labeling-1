from typing import Dict

from .netxtypes import *


# ---------------------------------------------------------------------------
# read_vecsv
# write_vecsv
# ---------------------------------------------------------------------------
# -vertices.csv
# -edges.csv
def _vecsv_strip(path: str) -> str:
    suffix = "-vertices.csv"
    if path.endswith(suffix):
        return path[0:-len(suffix)]
    suffix = "-edges.csv"
    if path.endswith(suffix):
        return path[0:-len(suffix)]
    suffix = ".csv"
    if path.endswith(suffix):
        return path[0:-len(suffix)]
    else:
        return path
# end


def load_vecsv(path: str, comments="#", header=True, separator=",", create_using=None) -> AGraph:
    return read_vecsv(path, comments, header, separator, create_using)


def read_vecsv(path: str, comments="#", header=True, separator=",", create_using=None) -> AGraph:
    """
    Read a graph saved in two files: "-vertices.csv" and "-edges.csv"

    :param path:
    :param comments:
    :param header:
    :param separator:
    :param create_using:
    :return:
    """

    def parse(s):
        try:
            return int(s)
        except:
            pass
        try:
            return float(s)
        except:
            pass
        return s
    # end

    vfile = _vecsv_strip(path) + "-vertices.csv"
    efile = _vecsv_strip(path) + "-edges.csv"

    if create_using is not None:
        g = create_using
    else:
        g = nx.Graph()

    # read nodes
    idmap: Dict[int, int] = dict()
    columns = None
    with open(vfile) as vfin:
        for line in vfin:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(comments):
                continue
            if columns is None:
                if header:
                    columns = line.split(separator)
                    continue
                else:
                    ncols = len(line.split(separator))
                    columns = [f"c{i+1:02}" for i in range(ncols)]
            # end
            props = list(map(parse, line.split(separator)))

            node = len(idmap)
            idmap[props[0]] = node

            # node = props[0]
            nattrs = dict()
            for i in range(0, len(columns)):
                nattrs[columns[i]] = props[i]

            g.add_node(node, **nattrs)
        # end

    # read edges
    columns = None
    with open(efile) as efin:
        for line in efin:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(comments):
                continue
            if columns is None:
                if header:
                    columns = line.split(separator)
                    continue
                else:
                    ncols = len(line.split(separator))
                    columns = [f"c{i + 1:02}" for i in range(ncols)]
            # end
            edge = list(map(parse, line.split(separator)))
            source = idmap[edge[0]]
            target = idmap[edge[1]]

            eattrs = dict()
            for i in range(0, len(columns)):
                eattrs[columns[i]] = props[i]

            g.add_edge(source, target, **eattrs)
        # end
    return g
# end


def save_vecsv(g: AGraph, path: str):
    write_vecsv(g, path)


def write_vecsv(g: AGraph, path: str):

    vfile = _vecsv_strip(path) + "-vertices.csv"
    efile = _vecsv_strip(path) + "-edges.csv"

    # write nodes
    with open(vfile, mode="w") as wrt:
        header = list(map(str, ["id"] + list(g.nodes[0].keys())))
        wrt.write(",".join(header) + "\n")

        for n in g.nodes:
            data = list(map(str, [n] + list(g.nodes[n].values())))
            wrt.write(",".join(data) + "\n")
        # end
    # end

    # write edges
    with open(efile, mode="w") as wrt:
        wrt.write("# source,target : direct graph\n")
        wrt.write("# id1,id2 : simple graph\n")
        if g.is_directed():
            wrt.write("source,target\n")
        else:
            wrt.write("id1,id2\n")

        for e in g.edges:
            u, v = e
            wrt.write(f"{u},{v}\n")
    # end
# end


# ---------------------------------------------------------------------------
# read_odem
# ---------------------------------------------------------------------------

def read_odem(path: str, create_using=None) -> AGraph:
    """
    Read the content of an "odem" file:
    the elements 'type' are nodes with attributes

        - name
        - classification
        - namespace:  based on the parent node 'namespace'

    the elements 'depends-on' are used to create directed edges, with attribute

        - classification

    :param path: path of the ODEM file
    :param create_using: the graph to populate, or None
    :return: the populated graph
    """
    if create_using is not None:
        g = create_using
    else:
        g = nx.Graph()

    from xml.etree import ElementTree
    document = ElementTree.parse(path)

    # name -> id map
    nameId = dict()

    containers = document.findall("./*/container")
    for container in containers:
        # scan for nodes
        namespaces = container.findall("namespace")
        for namespace in namespaces:
            nspace = namespace.attrib['name']
            for type in namespace.iter('type'):
                name = type.attrib['name']
                classification = type.attrib['classification']
                visibility = type.attrib['visibility']

                nid = len(nameId)
                nameId[name] = nid

                g.add_node(nid, name=name, classification=classification, namespace=nspace)
            # end
        # end

    # scan for edges
    containers = document.findall("./*/container")
    for container in containers:
        namespaces = container.findall("namespace")
        for namespace in namespaces:
            for type in namespace.iter('type'):
                sourceName = type.attrib['name']
                sourceId = nameId[sourceName]
                for dependencies in type.iter('dependencies'):
                    for dependsOn in dependencies.iter('depends-on'):
                        targetName = dependsOn.attrib['name']
                        classification = dependsOn.attrib['classification']
                        # skip edges with missing nodes
                        if targetName not in nameId:
                            continue
                        targetId = nameId[targetName]
                        g.add_edge(sourceId, targetId, classification=classification)
                    # end
                # end
            # end
        # end
    # end
    return g
# end


