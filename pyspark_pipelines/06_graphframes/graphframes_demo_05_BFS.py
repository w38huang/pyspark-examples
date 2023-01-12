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

    vertices = spark.createDataFrame([("BLR", "Bangalore", 20),
                                      ("MUM", "Mumbai", 25),
                                      ("AHM", "Ahmedabad", 4),
                                      ("CHN", "Chennai", 21),
                                      ("DEL", "Delhi", 23),
                                      ("HYD", "Hyderabad", 18),
                                      ("MYS", "Mysore", 2)], ["id", "city", "population"])
    # vertices.show(10, False)

    edges = spark.createDataFrame([("BLR", "MUM", 2000),
                                   ("AHM", "BLR", 3000),
                                   ("AHM", "MUM", 800),
                                   ("CHN", "MYS", 600),
                                   ("HYD", "MUM", 2300),
                                   ("MUM", "CHN", 2800),
                                   ("DEL", "HYD", 2600),
                                   ("BLR", "DEL", 3200),
                                   ("HYD", "BLR", 1000),
                                   ("BLR", "HYD", 1000),
                                   ("BLR", "MYS", 300),
                                   ("DEL", "CHN", 1900),
                                   ("MUM", "BLR", 2000),
                                   ("AHM", "DEL", 1700)], ["src", "dst", "distance"])

    ## step 1: # Below will show an error because in vertices id column should be there but in our case it is airport_id.
    ## Same like this in edges "src" and "dst" column should be there
    ## flights_route = GraphFrame(vertices, edges)

    ## step 2: rename vertices as id and construct graphframe

    cities = GraphFrame(vertices, edges)
    plot_graph(cities)

    paths = cities.bfs("id = 'AHM'", "population > 20")
    paths = cities.bfs("id = 'CHN'", "population < 10")

    paths = cities.bfs(fromExpr="city = 'Ahmedabad'",
                       toExpr="population > 10",
                       edgeFilter="distance < 2000")

    paths = cities.bfs(fromExpr="city = 'Bangalore'",
                       toExpr="population > 20",
                       maxPathLength=1)

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


def compute_airtime(cumulative_sum, edge):
    cumulative_sum = cumulative_sum + col(edge)['airtime']

    return cumulative_sum


def plot_graph(gf):
    gplot = nx.DiGraph()
    edge_labels = {}

    plt.figure(figsize=(6, 7))
    for node in gf.vertices.select('id').take(1000):
        gplot.add_node(node['id'])

    for row in gf.edges.select('src', 'dst', 'distance').take(1000):
        gplot.add_edge(row['src'], row['dst'])
        edge_labels[(row['src'], row['dst'])] = row['distance']

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



