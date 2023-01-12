import inspect
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

import networkx as nx
import matplotlib.pyplot as plt

from graphframes import GraphFrame
from graphframes.examples import *
# from graphframes.examples import Graphs


## cannot run since no data
def graphframes_demo_01():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    spark = SparkSession.builder.appName('GraphFramesExplore').getOrCreate()

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

    direct_flights = flight_routes.find("(a)-[e]->(b)")
    direct_flights = flight_routes.find("(source)-[edge]->(destination)")

    direct_flights.select('source.airport_name', 'destination.airport_name', 'edge.delay').show(10, False)
    direct_flights.show(10, False)

    ## edge is not shown
    direct_connection = flight_routes.find("(source)-[]->(destination)")
    direct_connection.show(10, False)
    direct_connection.filter('source.id = "SEA"').show(10, False)

    outgoing_flights = flight_routes.find("(source)-[edge]->()")
    outgoing_flights.show(10, False)

    outgoing_flights.groupby('source.airport_name').agg({'edge.delay': 'avg'}).show()

    ## incoming flights
    incoming_flights = flight_routes.find("()-[edge]->(destination)")
    incoming_flights.show()

    ## no flights
    no_flights = flight_routes.find("!(source)-[]->(destination)")

    flight_data = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("../../datasets/data/databricks/databricks-datasets/asa/airlines/2005.csv")

    flight_data.show(10, False)

    origins = flight_data.select('Origin').distinct().withColumnRenamed('Origin', 'City')
    destinations = flight_data.select('Dest').distinct().withColumnRenamed('Dest', 'City')
    vertices = origins.union(destinations).select('City').distinct().withColumnRenamed('City', 'id')

    routes = flight_data.select('Origin', 'Dest', 'AirTime', 'ArrDelay', 'DepDelay', 'Cancelled', 'Diverted')

    edges = routes.withColumnRenamed('Origin', 'src') \
        .withColumnRenamed('Dest', 'dst')
    us_flight_routes = GraphFrame(vertices, edges)

    us_direct_flights = us_flight_routes.find("(source)-[edge]->(destination)")
    us_direct_flights.filter('source.id = "PIT"').select('destination.id', 'edge.AirTime').show()
    us_direct_flights.filter('source.id = "SFO" and edge.ArrDelay < 30').select('destination.id', 'edge.ArrDelay').show()

    us_direct_flights.groupby('source.id', 'destination.id').sum('edge.Cancelled').show()

    one_hop_flights = us_flight_routes.find("(source)-[edge_1]->(destination_1); (destination_1)-[edge_2]->(final_destination)")
    one_hop_flights.filter('source.id != final_destination.id').show()
    return_flights = us_flight_routes.find("(source)-[edge_1]->(destination); (destination)-[edge_2]->(source)")

    return_flights.filter('edge_1.AirTime < 30 and edge_2.AirTime < 30') \
        .select('source.id', 'destination.id', 'edge_1.AirTime', 'edge_2.AirTime').show()

    no_return_flights = us_flight_routes.find("(source)-[edge_1]->(destination); !(destination)-[]->(source)")


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



