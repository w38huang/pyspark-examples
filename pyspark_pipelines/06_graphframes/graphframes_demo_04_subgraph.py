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

    vertices = spark.createDataFrame([("NYC", "New York City", 103),
                                      ("EUG", "Eugene", 65),
                                      ("AMW", "Ames", 35),
                                      ("AUS", "Austin", 46),
                                      ("BOS", "Boston", 57),
                                      ("SEA", "Seattle", 10),
                                      ("SFO", "San Francisco", 10),
                                      ("RDM", "Bend", 70),
                                      ("PDX", "Portland", 3)], ["id", "airport_name", "total_flights"])
    # vertices.show(10, False)

    edges = spark.createDataFrame([("NYC", "EUG", 33, 240),
                                   ("EUG", "AMW", -10, 100),
                                   ("AMW", "EUG", 0, 110),
                                   ("PDX", "AMW", 0, 120),
                                   ("BOS", "NYC", 44, 150),
                                   ("NYC", "BOS", 18, 160),
                                   ("NYC", "SFO", 9, 270),
                                   ("AUS", "PDX", -5, 180),
                                   ("BOS", "PDX", 3, 280),
                                   ("RDM", "PDX", -2, 80),
                                   ("SEA", "SFO", 5, 110),
                                   ("RDM", "SEA", 10, 150),
                                   ("SEA", "NYC", 35, 290),
                                   ("SFO", "RDM", 25, 300)], ["src", "dst", "delay", "airtime"])

    ## step 1: # Below will show an error because in vertices id column should be there but in our case it is airport_id.
    ## Same like this in edges "src" and "dst" column should be there
    ## flights_route = GraphFrame(vertices, edges)

    ## step 2: rename vertices as id and construct graphframe
    flight_routes = GraphFrame(vertices, edges)

    plot_directed_graph(flight_routes, 'delay')
    plot_directed_graph(flight_routes, 'airtime')

    not_busy_airports_short_flights = flight_routes.filterVertices('total_flights < 50') \
        .filterEdges('airtime < 200') \
        .dropIsolatedVertices()

    plot_directed_graph(not_busy_airports_short_flights, 'airtime')

    direct_routes = flight_routes.find("(source)-[edge]->(destination)")
    direct_routes.filter('edge.airtime > 120') \
        .filter('edge.delay > 10') \
        .filter('source.total_flights > destination.total_flights').show(5, False)

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



