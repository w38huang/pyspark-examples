import inspect
from functools import reduce

import matplotlib.pyplot as plt
import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from graphframes import GraphFrame
from graphframes.examples import Graphs

def graphframes_demo_03():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    spark = SparkSession.builder.appName('GraphFramesExplore').getOrCreate()

    spark.sparkContext.addPyFile('/Users/zhuohuawu/Documents/zw_progs/graphframes/graphframes-0.8.1-spark3.0-s_2.12.jar')

    vertices = spark.createDataFrame([("MHK", "Manhattan", 103),
                                      ("EUG", "Eugene", 65),
                                      ("AMW", "Ames", 35),
                                      ("STW", "Stowe", 2),
                                      ("SEA", "Seattle", 10),
                                      ("RDM", "Bend", 70),
                                      ("QTN", "Queenstown", 1),
                                      ("PDX", "Portland", 3)], ["airport_id", "airport_name", "total_flights"])
    # vertices.show(10, False)

    edges = spark.createDataFrame([("MHK", "EUG", 50),
                                   ("EUG", "AMW", -10),
                                   ("AMW", "EUG", 0),
                                   ("PDX", "AMW", 0),
                                   ("RDM", "PDX", -2),
                                   ("RDM", "SEA", 10),
                                   ("SEA", "MHK", 35),
                                   ("MHK", "RDM", 25)], ["src", "dst", "delay"])

    ## step 1: # Below will show an error because in vertices id column should be there but in our case it is airport_id.
    ## Same like this in edges "src" and "dst" column should be there
    ## flights_route = GraphFrame(vertices, edges)

    ## step 2: rename vertices as id and construct graphframe
    vertices = vertices.withColumnRenamed("airport_id", "id")
    flight_routes = GraphFrame(vertices, edges)

    plot_directed_graph(flight_routes, 'delay')
    plot_directed_graph(flight_routes, 'airtime')
    two_hop_routes = flight_routes.find("(a)-[edge_ab]->(b); (b)-[edge_bc]->(c); (c)-[edge_cd]->(d)")
    two_hop_routes.display()

    edges = ['edge_ab', 'edge_bc', 'edge_cd']

    total_airtime = reduce(compute_airtime, edges, lit(0))

    airtime_less_than_400 = two_hop_routes.withColumn("total_airtime", total_airtime).where(total_airtime < 400)
    airtime_less_than_400.select('a.id', 'edge_ab.delay', 'b.id', 'edge_bc.delay', 'c.id', 'edge_cd.delay',
                                 'd.id').show()

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


def compute_airtime(cumulative_sum, edge):
    cumulative_sum = cumulative_sum + col(edge)['airtime']

    return cumulative_sum


def plot_directed_graph(gf, relationship):
    gplot = nx.DiGraph()
    edge_labels = {}

    plt.figure(figsize=(6, 7))
    for node in gf.vertices.select('id').take(1000):
        gplot.add_node(node['id'])

    for row in gf.edges.select('src', 'dst', relationship).take(1000):
        gplot.add_edge(row['src'], row['dst'])
        edge_labels[(row['src'], row['dst'])] = row[relationship]

    pos = nx.shell_layout(gplot)

    nx.draw(gplot,
            pos,
            with_labels=True,
            font_weight='bold',
            node_size=1800,
            font_size=15,
            width=2)

    nx.draw_networkx_edge_labels(gplot,
                                 pos,
                                 edge_labels=edge_labels,
                                 font_color='green',
                                 font_size=10,
                                 font_weight='bold')

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    graphframes_demo_03()

# coding: utf-8



