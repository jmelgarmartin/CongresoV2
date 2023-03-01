from datetime import datetime
from py2neo import Graph
import pickle
import pandas as pd
import argparse
from diccionarios_auxiliares import dict_abreviaturas_grupos
from os import getcwd


def conecta_neo4j():
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "congreso"))
    print(str(datetime.now()) + ' ' + 'INICIO Indices')

    graph.run('CREATE TEXT INDEX I_DICE_grupo IF NOT EXISTS FOR ()-[r:DICE]-() ON (r.grupo_parlamentario)')
    graph.run('CREATE TEXT INDEX I_DICE_fecha IF NOT EXISTS FOR ()-[r:DICE]-() ON (r.fecha)')
    graph.run('CREATE TEXT INDEX I_DISCURSO_grupo IF NOT EXISTS FOR ()-[r:DISCURSO]-() ON (r.grupo_parlamentario)')
    graph.run('CREATE TEXT INDEX I_DISCURSO_fecha IF NOT EXISTS FOR ()-[r:DISCURSO]-() ON (r.fecha)')

    print(str(datetime.now()) + ' ' + 'FIN Indices')
    return graph


def borra_dubgrafos(graph):
    df_subgrafos = graph.run("call gds.graph.list() YIELD graphName RETURN graphName").to_data_frame()
    for elem in list(df_subgrafos['graphName']):
        graph.run(f"call gds.graph.drop('{elem}')")


def genera_subgrafos_global(graph):
    tx = graph.begin()
    tx.run("CALL gds.graph.project.cypher( \
      'DISCURSO_GLOBAL', \
      'MATCH (n:Palabra) RETURN id(n) AS id, labels(n) as labels', \
      'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
       RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")

    tx.run("CALL gds.graph.project.cypher( \
      'DICE_GLOBAL', \
      'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
      'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
      RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")
    graph.commit(tx)


def lanza_algoritmos_global(graph):
    tx = graph.begin()
    graph.run('CREATE INDEX I_COM_DISCURSO_GLOBAL IF NOT EXISTS FOR (p:Palabra) ON (p.COM_DISCURSO_GLOBAL)')
    graph.commit(tx)
    tx = graph.begin()

    # extraemos los datos a csv para procesarlos después
    tx.run(f"CALL{{ \
        CALL gds.pageRank.stream('DISCURSO_GLOBAL') \
        YIELD nodeId, score \
        RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
        ORDER BY pr DESC, palabra \
        LIMIT 100 \
    }} WITH palabra, pr \
    MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
    WHERE p.lemma = palabra \
    RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
    p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
        .to_csv(getcwd() + "\\resultados_analisis\\PR_DISCURSO_GLOBAL.csv", index=False, header=True)

    tx.run("CALL  gds.louvain.write('DISCURSO_GLOBAL', { \
            writeProperty: 'COM_DISCURSO_GLOBAL',\
            includeIntermediateCommunities: false, \
            maxIterations: 1000}) \
            YIELD communityCount, modularity, modularities")

    tx.run(f"CALL{{ \
        CALL gds.pageRank.stream('DICE_GLOBAL') \
        YIELD nodeId, score \
        RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
        ORDER BY pr DESC, palabra \
        LIMIT 100 \
        }} WITH palabra, pr \
        MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
        WHERE p.lemma = palabra \
        RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
        p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
        .to_csv(getcwd() + "\\resultados_analisis\\PR_DICE_GLOBAL.csv", index=False, header=True)

    tx.run(f"CALL {{ \
            MATCH(p:Palabra) \
            RETURN p.COM_DISCURSO_GLOBAL as comunidad, count(p.lemma) as numero \
            ORDER BY numero desc \
            LIMIT 10 \
            }} WITH comunidad \
            MATCH (p:Palabra) \
            WHERE p.COM_DISCURSO_GLOBAL = comunidad \
            RETURN p.lemma as palabra, p.polaridad_round as polaridad, p.COM_DISCURSO_GLOBAL as comunidad \
            ORDER BY comunidad, palabra").to_data_frame() \
        .to_csv(getcwd() + "\\resultados_analisis\\COM_DISCURSO_GLOBAL.csv", index=False, header=True)

    # borramos las variables generadas
    tx.run(f"MATCH (p) REMOVE p.COM_DISCURSO_GLOBAL")

    graph.commit(tx)


def borra_subgrafos_global(graph):
    tx = graph.begin()
    tx.run("CALL gds.graph.drop('DISCURSO_GLOBAL')")
    tx.run("CALL gds.graph.drop('DICE_GLOBAL')")
    tx.run(f"DROP INDEX I_COM_DISCURSO_GLOBAL IF EXISTS")
    graph.commit(tx)


def genera_subgrafos_global_fecha(row):
    graph = row['graph']
    fecha_form = row['fecha']
    fecha = row['fecha'].replace('-', '_')
    tx = graph.begin()

    tx.run(f"CALL gds.graph.project.cypher( \
          'DISCURSO_{fecha}', \
          'MATCH (n:Palabra) RETURN id(n) AS id, labels(n) as labels', \
          'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
          WHERE d.fecha = \"{fecha_form}\" \
           RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
          YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")

    tx.run(f"CALL gds.graph.project.cypher( \
      'DICE_{fecha}', \
      'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
      'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
      WHERE d.fecha = \"{fecha_form}\" \
      RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels")

    graph.commit(tx)


def lanza_algoritmos_global_fecha(row):
    graph = row['graph']
    fecha_form = row['fecha']
    fecha = row['fecha'].replace('-', '_')

    tx = graph.begin()
    graph.run(f'CREATE INDEX I_COM_DISCURSO_{fecha} IF NOT EXISTS FOR (p:Palabra) ON (p.COM_DISCURSO_{fecha})')
    graph.commit(tx)
    tx = graph.begin()

    # extraemos los datos a csv para procesarlos después
    tx.run(f"CALL{{ \
            CALL gds.pageRank.stream('DISCURSO_{fecha}') \
            YIELD nodeId, score \
            RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
            ORDER BY pr DESC, palabra \
            LIMIT 100 \
        }} WITH palabra, pr \
        MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
        WHERE p.lemma = palabra \
        AND d.fecha = '{fecha_form}' \
        RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
        p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
        .to_csv(getcwd() + "\\resultados_analisis\\PR_DISCURSO_" + fecha + ".csv", index=False, header=True)

    tx.run(f"CALL  gds.louvain.write('DISCURSO_{fecha}', {{ \
               writeProperty: 'COM_DISCURSO_{fecha}',\
               includeIntermediateCommunities: false, \
               maxIterations: 1000}}) \
               YIELD communityCount, modularity, modularities")

    tx.run(f"CALL{{ \
        CALL gds.pageRank.stream('DICE_{fecha}') \
        YIELD nodeId, score \
        RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
        ORDER BY pr DESC, palabra \
        LIMIT 100 \
        }} WITH palabra, pr \
        MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
        WHERE p.lemma = palabra \
        AND d.fecha = '{fecha_form}' \
        RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
        p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
        .to_csv(getcwd() + "\\resultados_analisis\\PR_DICE_" + fecha + ".csv", index=False, header=True)

    tx.run(f"CALL {{ \
                MATCH(p:Palabra) \
                RETURN p.COM_DISCURSO_{fecha} as comunidad, count(p.lemma) as numero \
                ORDER BY numero desc \
                LIMIT 10 \
                }} WITH comunidad \
                MATCH (p:Palabra) \
                WHERE p.COM_DISCURSO_{fecha} = comunidad \
                RETURN p.lemma as palabra, p.polaridad_round as polaridad, p.COM_DISCURSO_{fecha} as comunidad \
                ORDER BY comunidad, palabra").to_data_frame() \
        .to_csv(getcwd() + "\\resultados_analisis\\COM_DISCURSO_" + fecha + ".csv", index=False, header=True)

    # borramos las variables generadas
    tx.run(f"MATCH (p) REMOVE p.COM_DISCURSO_{fecha}")

    graph.commit(tx)


def borra_subgrafos_global_fecha(row):
    graph = row['graph']
    fecha = row['fecha'].replace('-', '_')
    tx = graph.begin()

    tx.run(f"CALL gds.graph.drop('DISCURSO_{fecha}')")
    tx.run(f"CALL gds.graph.drop('DICE_{fecha}')")
    tx.run(f"DROP INDEX I_COM_DISCURSO_{fecha} IF EXISTS")

    graph.commit(tx)


def genera_subgrafos_global_grupo(row):
    graph = row['graph']
    grupo = row['grupo parlamentario']
    abreb = row['abreb'].replace('-', '_').replace('(', '').replace(')', '')

    if abreb != 'N/A':
        tx = graph.begin()
        tx.run(f"CALL gds.graph.project.cypher( \
                      'DISCURSO_{abreb}', \
                      'MATCH (n:Palabra) RETURN id(n) AS id, labels(n) as labels', \
                      'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
                      WHERE d.grupo_parlamentario = \"{grupo}\" \
                       RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
                      YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, \
                      relationshipCount AS rels")

        tx.run(f"CALL gds.graph.project.cypher( \
                  'DICE_{abreb}', \
                  'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
                  'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
                  WHERE h.grupo_parlamentario = \"{grupo}\" \
                  RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
                  YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, \
                  relationshipCount AS rels")

        graph.commit(tx)


def lanza_algoritmos_global_grupo(row):
    graph = row['graph']
    grupo = row['grupo parlamentario']
    abreb = row['abreb'].replace('-', '_').replace('(', '').replace(')', '')

    if abreb != 'N/A':
        tx = graph.begin()
        graph.run(f'CREATE INDEX I_COM_DISCURSO_{abreb} IF NOT EXISTS FOR (p:Palabra) ON (p.COM_DISCURSO_{abreb})')
        graph.commit(tx)
        tx = graph.begin()
        # extraemos los datos a csv para procesarlos después
        tx.run(f"CALL{{ \
                    CALL gds.pageRank.stream('DISCURSO_{abreb}') \
                    YIELD nodeId, score \
                    RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
                    ORDER BY pr DESC, palabra \
                    LIMIT 100 \
                }} WITH palabra, pr \
                MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
                WHERE p.lemma = palabra \
                AND d.grupo_parlamentario = '{grupo}' \
                RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
                p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
            .to_csv(getcwd() + "\\resultados_analisis\\PR_DISCURSO_" + abreb + ".csv", index=False, header=True)

        tx.run(f"CALL  gds.louvain.write('DISCURSO_{abreb}', {{ \
                       writeProperty: 'COM_DISCURSO_{abreb}',\
                       includeIntermediateCommunities: false, \
                       maxIterations: 1000}}) \
                       YIELD communityCount, modularity, modularities")

        tx.run(f"CALL{{ \
                CALL gds.pageRank.stream('DICE_{abreb}') \
                YIELD nodeId, score \
                RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
                ORDER BY pr DESC, palabra \
                LIMIT 100 \
                }} WITH palabra, pr \
                MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
                WHERE p.lemma = palabra \
                AND d.grupo_parlamentario = '{grupo}' \
                RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
                p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
            .to_csv(getcwd() + "\\resultados_analisis\\PR_DICE_" + abreb + ".csv", index=False, header=True)

        tx.run(f"CALL {{ \
            MATCH(p:Palabra) \
            RETURN p.COM_DISCURSO_{abreb} as comunidad, count(p.lemma) as numero \
            ORDER BY numero desc \
            LIMIT 10 \
            }} WITH comunidad \
            MATCH (p:Palabra) \
            WHERE p.COM_DISCURSO_{abreb} = comunidad \
            RETURN p.lemma as palabra, p.polaridad_round as polaridad, p.COM_DISCURSO_{abreb} as comunidad \
            ORDER BY comunidad, palabra").to_data_frame() \
            .to_csv(getcwd() + "\\resultados_analisis\\COM_DISCURSO_" + abreb + ".csv", index=False, header=True)

        # borramos las variables generadas
        tx.run(f"MATCH (p) REMOVE p.COM_DISCURSO_{abreb}")

        graph.commit(tx)


def borra_subgrafos_global_grupo(row):
    graph = row['graph']
    abreb = row['abreb'].replace('-', '_').replace('(', '').replace(')', '')

    if abreb != 'N/A':
        tx = graph.begin()
        tx.run(f"CALL gds.graph.drop('DISCURSO_{abreb}')")
        tx.run(f"CALL gds.graph.drop('DICE_{abreb}')")
        tx.run(f"DROP INDEX I_COM_DISCURSO_{abreb} IF EXISTS")

        graph.commit(tx)


def genera_subgrafos_fecha_grupo(row):
    graph = row['graph']
    grupo = row['grupo parlamentario']
    fecha_form = row['fecha']
    fecha = row['fecha'].replace('-', '_')
    abreb = row['abreb'].replace('-', '_').replace('(', '').replace(')', '')

    if abreb != 'N/A':
        tx = graph.begin()
        graph.run(
            f'CREATE INDEX I_COM_DISCURSO_{fecha}_{abreb} \
             IF NOT EXISTS FOR (p:Palabra) ON (p.COM_DISCURSO_{fecha}_{abreb})')
        graph.commit(tx)
        tx = graph.begin()
        tx.run(f"CALL gds.graph.project.cypher( \
        'DISCURSO_{fecha}_{abreb}', \
        'MATCH (n:Palabra) RETURN id(n) AS id, labels(n) as labels', \
        'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) \
        WHERE d.grupo_parlamentario = \"{grupo}\" \
        AND d.fecha = \"{fecha_form}\" \
        RETURN id(p1) AS source, id(p2) AS target, type(d) AS type') \
        YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, \
        relationshipCount AS rels")

        tx.run(f"CALL gds.graph.project.cypher( \
        'DICE_{fecha}_{abreb}', \
        'MATCH (n) RETURN id(n) AS id, labels(n) as labels', \
        'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
        WHERE d.grupo_parlamentario = \"{grupo}\" \
        AND d.fecha = \"{fecha_form}\" \
        RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces') \
        YIELD graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, \
        relationshipCount AS rels")

        graph.commit(tx)


def lanza_algoritmos_fecha_grupo(row):
    graph = row['graph']
    fecha = row['fecha'].replace('-', '_')
    abreb = row['abreb'].replace('-', '_').replace('(', '').replace(')', '')
    grupo = row['grupo parlamentario']
    fecha_form = row['fecha']

    if abreb != 'N/A':
        tx = graph.begin()
        # extraemos los datos a csv para procesarlos después
        tx.run(f"CALL{{ \
        CALL gds.pageRank.stream('DISCURSO_{fecha}_{abreb}') \
        YIELD nodeId, score \
        RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
        ORDER BY pr DESC, palabra \
        LIMIT 100 \
        }} WITH palabra, pr \
        MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
        WHERE p.lemma = palabra \
        AND d.grupo_parlamentario = '{grupo}' \
        AND d.fecha = \"{fecha_form}\" \
        RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
        p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
            .to_csv(getcwd() + "\\resultados_analisis\\PR_DISCURSO_" + fecha + "_" + abreb + ".csv", index=False,
                    header=True)

        tx.run(f"CALL  gds.louvain.write('DISCURSO_{fecha}_{abreb}', {{ \
        writeProperty: 'COM_DISCURSO_{fecha}_{abreb}',\
        includeIntermediateCommunities: false, \
        maxIterations: 1000}}) \
        YIELD communityCount, modularity, modularities")

        tx.run(f"CALL{{ \
        CALL gds.pageRank.stream('DICE_{fecha}_{abreb}') \
        YIELD nodeId, score \
        RETURN gds.util.asNode(nodeId).lemma AS palabra, score as pr \
        ORDER BY pr DESC, palabra \
        LIMIT 100 \
        }} WITH palabra, pr \
        MATCH (h:Hablante)-[d:DICE]->(p:Palabra) \
        WHERE p.lemma = palabra \
        AND d.grupo_parlamentario = '{grupo}' \
        AND d.fecha = \"{fecha_form}\" \
        RETURN h.grupo_parlamentario as grupo_parlamentario, sum(d.veces) as veces,\
        p.lemma as palabra, d.fecha as fecha, pr as total").to_data_frame() \
            .to_csv(getcwd() + "\\resultados_analisis\\PR_DICE_" + fecha + "_" + abreb + ".csv", index=False,
                    header=True)

        tx.run(f"CALL {{ \
                MATCH(p:Palabra) \
                RETURN p.COM_DISCURSO_{fecha}_{abreb} as comunidad, count(p.lemma) as numero \
                ORDER BY numero desc \
                LIMIT 10 \
                }} WITH comunidad \
                MATCH (p:Palabra) \
                WHERE p.COM_DISCURSO_{fecha}_{abreb} = comunidad \
                RETURN p.lemma as palabra, p.polaridad_round as polaridad, p.COM_DISCURSO_{fecha}_{abreb} as comunidad \
                ORDER BY comunidad, palabra").to_data_frame() \
            .to_csv(getcwd() + "\\resultados_analisis\\COM_DISCURSO_" + fecha + "_" + abreb + ".csv", index=False,
                    header=True)

        # borramos las variables generadas
        tx.run(f"MATCH (p) REMOVE p.COM_DISCURSO_{fecha}_{abreb}")

        graph.commit(tx)


def borra_subgrafos_fecha_grupo(row):
    graph = row['graph']
    fecha = row['fecha'].replace('-', '_')
    abreb = row['abreb'].replace('-', '_').replace('(', '').replace(')', '')

    if abreb != 'N/A':
        tx = graph.begin()
        tx.run(f"CALL gds.graph.drop('DISCURSO_{fecha}_{abreb}')")
        tx.run(f"CALL gds.graph.drop('DICE_{fecha}_{abreb}')")
        tx.run(f"DROP INDEX I_COM_DISCURSO_{fecha}_{abreb} IF EXISTS")

        graph.commit(tx)


def generar_subgrafos_y_ejecutar_algoritmos(graph, df_lemmas):
    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos')
    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos global')

    genera_subgrafos_global(graph)
    lanza_algoritmos_global(graph)
    borra_subgrafos_global(graph)

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos global')

    df_fechas = pd.DataFrame(df_lemmas['fecha'].unique(), columns=['fecha'])
    df_grupos = pd.DataFrame(df_lemmas['grupo parlamentario'].unique(), columns=['grupo parlamentario'])
    df_combinado = pd.merge(df_fechas, df_grupos, how='cross')

    buscar_abreviacion = lambda x: dict_abreviaturas_grupos.get(x, 'N/A')

    df_grupos['abreb'] = df_grupos['grupo parlamentario'].apply(buscar_abreviacion)
    df_combinado['abreb'] = df_combinado['grupo parlamentario'].apply(buscar_abreviacion)

    # eliminamos aquellos registros que no contengan un grupo con abreb
    df_grupos = df_grupos[df_grupos['abreb'] != 'N/A']
    df_combinado = df_combinado[df_combinado['abreb'] != 'N/A']

    df_fechas['graph'] = graph
    df_grupos['graph'] = graph
    df_combinado['graph'] = graph

    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos - fecha')
    print(len(df_fechas))
    for i in range(0, len(df_fechas), 10):
        if (i % 100) == 0:
            print(str(datetime.now()) + ' ' + str(i))
        block = df_fechas[i:i + 10]
        block.apply(genera_subgrafos_global_fecha, axis=1)
        block.apply(lanza_algoritmos_global_fecha, axis=1)
        block.apply(borra_subgrafos_global_fecha, axis=1)

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos - fecha')

    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos - grupo')
    print(len(df_grupos))
    for i in range(0, len(df_grupos), 10):
        if (i % 100) == 0:
            print(str(datetime.now()) + ' ' + str(i))
        block = df_grupos[i:i + 10]
        block.apply(genera_subgrafos_global_grupo, axis=1)
        block.apply(lanza_algoritmos_global_grupo, axis=1)
        block.apply(borra_subgrafos_global_grupo, axis=1)

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos - grupo')

    print(str(datetime.now()) + ' ' + 'INICIO generar subgrafos y algoritmos - grupo/fecha')
    print(len(df_combinado))
    for i in range(0, len(df_combinado), 10):
        if (i % 100) == 0:
            print(str(datetime.now()) + ' ' + str(i))
        block = df_combinado[i:i + 10]
        block.apply(genera_subgrafos_fecha_grupo, axis=1)
        block.apply(lanza_algoritmos_fecha_grupo, axis=1)
        block.apply(borra_subgrafos_fecha_grupo, axis=1)

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos - grupo/fecha')

    print(str(datetime.now()) + ' ' + 'FIN generar subgrafos y algoritmos')


def main(borrar):
    print(str(datetime.now()) + ' ' + 'INICIO PROCESO')
    graph = conecta_neo4j()
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
