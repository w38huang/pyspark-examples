import inspect
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

import networkx as nx
import matplotlib.pyplot as plt

from graphframes import GraphFrame
from graphframes.examples import *
# from graphframes.examples import Graphs

def graphframes_demo_01():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    spark = SparkSession.builder.appName('GraphFramesExplore').getOrCreate()

    # add jar in pycharm and below is no need
    # spark.sparkContext.addPyFile('/Users/zhuohuawu/Documents/zw_progs/graphframes/graphframes-0.8.1-spark3.0-s_2.12.jar')

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

    ## step 3: show vertices, edges, inDegrees
    print("flight_routes: vertices: ")
    flight_routes.vertices.show(10, False)

    print("flight_routes: edges: ")
    flight_routes.edges.show(10, False)

    ## select src and dst pair
    flight_routes.edges.select("src", "dst").show()
    plot_directed_graph(flight_routes)

    ## indegree and outdegree
    flight_routes.inDegrees.show()
    flight_routes.outDegrees.show()
    flight_routes.degrees.show()
    flight_routes.inDegrees.filter('inDegree > 1').show(10, False)

    ## Low traffic airport
    low_traffic_airports = flight_routes.vertices.filter(flight_routes.vertices['total_flights'] < 30)
    low_traffic_airports.show(3, False)

    low_traffic_airports_graph = flight_routes.filterVertices('total_flights < 30')
    plot_directed_graph(low_traffic_airports_graph)


    ## Showing airport with max operational flights
    max_operational_flights = flight_routes.vertices.groupBy().agg(max("total_flights").alias("max_flights"))
    max_operational_flights.show(3, False)

    max_flights_airport = flight_routes.vertices.filter("total_flights = {0}".format(max_operational_flights.first()['max_flights']))
    max_flights_airport.show(3, False)

    max_flights_graph = flight_routes.filterVertices("total_flights = {0}".format(max_operational_flights.first()['max_flights']))
    plot_directed_graph(max_flights_graph)

    ## Displaying delay and destination airport from airport which operates max flights
    MHK_flights = flight_routes.edges.filter((col("src") == max_flights_airport.first()["id"]) | (col("dst") == max_flights_airport.first()["id"]))
    MHK_flights.show(3, False)

    MHK_flights_graph = flight_routes.filterEdges((col("src") == max_flights_airport.first()["id"]) | (col("dst") == max_flights_airport.first()["id"]))
    plot_directed_graph(MHK_flights_graph)


    g = Graphs(spark).friends()
    g.vertices.show(3, False)
    g.edges.show()
    g.inDegrees.show()
    g.outDegrees.show()

    num_follows_graph = g.filterEdges("relationship = 'follow'")
    num_follows_graph.edges.show()

    print(g)

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


def plot_directed_graph(gf):
    gplot = nx.DiGraph()
    edge_labels = {}

    plt.figure(figsize=(6, 7))
    for node in gf.vertices.select('id').take(1000):
        gplot.add_node(node['id'])

    for row in gf.edges.select('src', 'dst', 'delay').take(1000):
        gplot.add_edge(row['src'], row['dst'])
        edge_labels[(row['src'], row['dst'])] = row['delay']

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
    graphframes_demo_01()

# coding: utf-8



