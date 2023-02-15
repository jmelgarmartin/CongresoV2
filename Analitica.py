from datetime import datetime
from py2neo import Graph
import pickle
import pandas as pd
import argparse
from diccionarios_auxiliares import dict_abreviaturas_grupos


def borra_dubgrafos(graph):
    df_subgrafos = graph.run("call gds.graph.list() YIELD graphName RETURN graphName").to_data_frame()
    for elem in list(df_subgrafos['graphName']):
        graph.run(f"call gds.graph.drop('{elem}')")


def genera_subgrafos_global(graph):
    graph.run("CALL gds.graph.project.cypher( \
      'DISCURSO_GLOBAL', \
      'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
      'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
       RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")

    graph.run("CALL gds.graph.project.cypher( \
      'DICE_GLOBAL', \
      'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
      'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
      RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")


def lanza_algoritmos_global(graph):
    graph.run("CALL gds.pageRank.write('DISCURSO_GLOBAL', { \
        maxIterations: 20, \
        dampingFactor: 0.85, \
        writeProperty: 'PR_DISCURSO_GLOBAL'}) \
        YIELD nodePropertiesWritten, ranIterations")

    graph.run("CALL  gds.louvain.write('DISCURSO_GLOBAL', { \
        writeProperty: 'COM_DISCURSO_GLOBAL',\
        includeIntermediateCommunities: false, \
        maxIterations: 1000}) \
        YIELD communityCount, modularity, modularities")

    graph.run("CALL gds.pageRank.write('DICE_GLOBAL', { \
        maxIterations: 20, \
        dampingFactor: 0.85, \
        writeProperty: 'PR_DICE_GLOBAL',\
        relationshipWeightProperty: 'veces'}) \
        YIELD nodePropertiesWritten, ranIterations")


def genera_subgrafos_global_fecha(row):
    graph = row['graph']
    fecha = row['fecha']

    graph.run(f"CALL gds.graph.project.cypher( \
          'DISCURSO_{fecha}', \
          'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
          'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
          WHERE d.fecha = \"{fecha} \" \
           RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
          YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")

    graph.run(f"CALL gds.graph.project.cypher( \
      'DICE_{fecha}', \
      'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
      'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
      WHERE d.fecha = \"{fecha} \" \
      RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")


def lanza_algoritmos_global_fecha(row):
    graph = row['graph']
    fecha = row['fecha']

    graph.run(f"CALL gds.pageRank.write('DISCURSO_{fecha}', {{ \
        maxIterations: 20, \
        dampingFactor: 0.85, \
        writeProperty: 'PR_DISCURSO_{fecha}'}}) \
        YIELD nodePropertiesWritten, ranIterations")

    graph.run(f"CALL  gds.louvain.write('DISCURSO_{fecha}', {{ \
        writeProperty: 'COM_DISCURSO_{fecha}',\
        includeIntermediateCommunities: false, \
        maxIterations: 1000}}) \
        YIELD communityCount, modularity, modularities")

    graph.run(f"CALL gds.pageRank.write('DICE_{fecha}', {{ \
        maxIterations: 20, \
        dampingFactor: 0.85, \
        writeProperty: 'PR_DICE_{fecha}',\
        relationshipWeightProperty: 'veces'}}) \
        YIELD nodePropertiesWritten, ranIterations")


def genera_subgrafos_global_grupo(row):
    graph = row['graph']
    grupo = row['grupo']
    abreb = row['abreb']

    if abreb != 'N/A':
        graph.run(f"CALL gds.graph.project.cypher( \
              'DISCURSO_{abreb}', \
              'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
              'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
              WHERE d.grupo_parlamentario = \"{grupo} \" \
               RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
              YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")

        graph.run(f"CALL gds.graph.project.cypher( \
          'DICE_{abreb}', \
          'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
          'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
          WHERE d.grupo_parlamentario = \"{grupo} \" \
          RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
          YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")


def lanza_algoritmos_global_grupo(row):
    graph = row['graph']
    abreb = row['abreb']

    if abreb != 'N/A':
        graph.run(f"CALL gds.pageRank.write('DISCURSO_{abreb}', {{ \
            maxIterations: 20, \
            dampingFactor: 0.85, \
            writeProperty: 'PR_DISCURSO_{abreb}'}}) \
            YIELD nodePropertiesWritten, ranIterations")

        graph.run(f"CALL  gds.louvain.write('DISCURSO_{abreb}', {{ \
            writeProperty: 'COM_DISCURSO_{abreb}',\
            includeIntermediateCommunities: false, \
            maxIterations: 1000}}) \
            YIELD communityCount, modularity, modularities")

        graph.run(f"CALL gds.pageRank.write('DICE_{abreb}', {{ \
            maxIterations: 20, \
            dampingFactor: 0.85, \
            writeProperty: 'PR_DICE_{abreb}',\
            relationshipWeightProperty: 'veces'}}) \
            YIELD nodePropertiesWritten, ranIterations")


def genera_subgrafos_fecha_grupo(row):
    graph = row['graph']
    grupo = row['grupo']
    fecha = row['fecha']
    abreb = row['abreb']

    if abreb != 'N/A':
        graph.run(f"CALL gds.graph.project.cypher( \
                      'DISCURSO_{fecha}_{abreb}', \
                      'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
                      'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
                      WHERE d.grupo_parlamentario = \"{grupo} \" \
                       AND d.fecha = \"{fecha} \" \
                       RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
                      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, \
                       relationshipCount AS rels")

        graph.run(f"CALL gds.graph.project.cypher( \
                  'DICE_{fecha}_{abreb}', \
                  'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
                  'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
                  WHERE d.grupo_parlamentario = \"{grupo} \" \
                  AND d.fecha = \"{fecha} \" \
                  RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
                  YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, \
                   relationshipCount AS rels")


def lanza_algoritmos_fecha_grupo(row):
    graph = row['graph']
    fecha = row['fecha']
    abreb = row['abreb']

    if abreb != 'N/A':
        graph.run(f"CALL gds.pageRank.write('DISCURSO_{fecha}_{abreb}', {{ \
                maxIterations: 20, \
                dampingFactor: 0.85, \
                writeProperty: 'PR_DISCURSO_{fecha}_{abreb}'}}) \
                YIELD nodePropertiesWritten, ranIterations")

        graph.run(f"CALL  gds.louvain.write('DISCURSO_{fecha}_{abreb}', {{ \
                writeProperty: 'COM_DISCURSO_{fecha}_{abreb}',\
                includeIntermediateCommunities: false, \
                maxIterations: 1000}}) \
                YIELD communityCount, modularity, modularities")

        graph.run(f"CALL gds.pageRank.write('DICE_{fecha}_{abreb}', {{ \
                maxIterations: 20, \
                dampingFactor: 0.85, \
                writeProperty: 'PR_DICE_{fecha}_{abreb}',\
                relationshipWeightProperty: 'veces'}}) \
                YIELD nodePropertiesWritten, ranIterations")


def generar_subgrafos_y_ejecutar_algoritmos(graph, df_lemmas):
    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos')

    genera_subgrafos_global(graph)
    lanza_algoritmos_global(graph)

    df_fechas = pd.DataFrame(df_lemmas['fecha'].unique(), columns=['fecha'])
    df_grupos = pd.DataFrame(df_lemmas['grupo parlamentario'].unique(), columns=['grupo parlamentario'])
    df_combinado = pd.merge(df_fechas, df_grupos, how='cross')

    buscar_abreviacion = lambda x: dict_abreviaturas_grupos.get(x, 'N/A')

    df_grupos['abreb'] = df_grupos['grupo parlamentario'].apply(buscar_abreviacion)
    df_combinado['abreb'] = df_combinado['grupo parlamentario'].apply(buscar_abreviacion)

    df_fechas['graph'] = graph
    df_grupos['graph'] = graph
    df_combinado['graph'] = graph

    total_subgrafos = 2

    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos - fecha')

    for i in range(0, len(df_fechas), 1000):
        block = df_fechas[i:i + 1000]
        block.apply(genera_subgrafos_global_fecha, axis=1)
        block.apply(lanza_algoritmos_global_fecha, axis=1)

    subgrafos_generados = \
        graph.run("call gds.graph.list() YIELD graphName RETURN COUNT(graphName) as numero").to_series()[0]
    total_subgrafos = total_subgrafos + (len(df_fechas) * 2)

    if total_subgrafos != subgrafos_generados:
        print("ERRROR NO SE HA GENERADO TODO: ")
        print("Se debía generar: " + str(total_subgrafos))
        print("Se han generado: " + str(subgrafos_generados))

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos - fecha')

    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos - grupo')

    for i in range(0, len(df_grupos), 1000):
        block = df_grupos[i:i + 1000]
        block.apply(genera_subgrafos_global_grupo, axis=1)
        block.apply(lanza_algoritmos_global_grupo, axis=1)

    total_subgrafos = total_subgrafos + (len(df_grupos) * 2)

    if total_subgrafos != subgrafos_generados:
        print("ERRROR NO SE HA GENERADO TODO: ")
        print("Se debía generar: " + str(total_subgrafos))
        print("Se han generado: " + str(subgrafos_generados))

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos - grupo')

    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos - grupo/fecha')

    for i in range(0, len(df_combinado), 1000):
        block = df_combinado[i:i + 1000]
        block.apply(genera_subgrafos_fecha_grupo, axis=1)
        block.apply(lanza_algoritmos_fecha_grupo, axis=1)

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos - grupo/fecha')

    subgrafos_generados = \
        graph.run("call gds.graph.list() YIELD graphName RETURN COUNT(graphName) as numero").to_series()[0]

    total_subgrafos = total_subgrafos + (len(df_combinado) * 2)

    if total_subgrafos != subgrafos_generados:
        print("ERRROR NO SE HA GENERADO TODO: ")
        print("Se debía generar: " + str(total_subgrafos))
        print("Se han generado: " + str(subgrafos_generados))

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos')


def main(borrar):
    print(str(datetime.now()) + ' ' + 'INICIO PROCESO')
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "congreso"))
    palabras_lemm = 'palabras.pickle'

    if borrar:
        borra_dubgrafos(graph)
    else:
        print(str(datetime.now()) + ' ' + 'Cargando Lemmas')
        with open(palabras_lemm, "rb") as f:
            df_lemmas = pickle.load(f)
        print(str(datetime.now()) + ' ' + 'Lemmas cargados')

        generar_subgrafos_y_ejecutar_algoritmos(graph, df_lemmas)

    print(str(datetime.now()) + ' ' + 'FIN PROCESO')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análisis en Neo4j')
    parser.add_argument('borrado', help='Vale True si se quieren borrar los subgrafos')
    args = parser.parse_args()
    if args.borrado == 'True':
        main(True)
    else:
        main(False)


