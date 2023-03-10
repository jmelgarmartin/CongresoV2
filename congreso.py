from PyPDF2 import PdfReader
import re
import string
from os import scandir, getcwd
from datetime import datetime
import spacy
import pandas as pd
import concurrent.futures
import pickle
import os
from textblob import TextBlob
from translate import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from py2neo import Graph
from diccionarios_auxiliares import correccion_inicio, excepciones_texto, excluidos, dict_hablantes, \
    documentos_sin_dialogos
import textstat
import dask.dataframe as dd


def conecta_neo4j():
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "congreso"))

    graph.run('CREATE TEXT INDEX I_Hablante IF NOT EXISTS FOR (h:Hablante) ON (h.nombre)')
    graph.run('CREATE TEXT INDEX I_Palabra IF NOT EXISTS FOR (p:Palabra) ON (p.lemma)')

    return graph


def ls(ruta=getcwd() + '\\documentos'):
    return [arch.path for arch in scandir(ruta) if arch.is_file()]


def extrae_paginas(reader):
    paginas = []
    # extraemos las páginas y eliminamos las cabeceras, y conservamos la fecha del debate
    for i in range(len(reader.pages)):
        if i == 0:
            paginas.extend(reader.pages[i].extract_text().split('\n')[9:])
        else:
            if i == 1:
                fecha_proceso = reader.pages[i].extract_text().split('\n')[2]
            paginas.extend(reader.pages[i].extract_text().split('\n')[3:])
    return paginas, fecha_proceso


def procesa_paginas(paginas, documento):
    paginas_procesadas = []
    linea_aux = ''
    for linea in paginas:
        if linea.startswith('cve: DSCD'):
            continue
        else:
            linea_limpia = linea.replace('\u2002', ' ').replace('.', '').replace(chr(8230), '')
            linea_aux = linea_aux + linea_limpia
            if (len(linea_limpia) == 0) or linea_limpia[-1] != ' ':
                # añado una excepcion, se ha detectado que en un documento falta el símbolo ':' después de un hablante
                if documento == r'C:\Users\jmelgar\Congreso\documentos\DSCD-14-PL-94.PDF' and \
                        linea_aux.startswith(
                            'El señor PRESIDENTE DEL GOBIERNO  (Sánchez Pérez-Castejón) La diferencia, señor Abascal, señorías de VOX,'):
                    linea_aux = linea_aux.replace('(Sánchez Pérez-Castejón) La diferencia,',
                                                  '(Sánchez Pérez-Castejón): La diferencia,')
                paginas_procesadas.append(linea_aux)
                linea_aux = ''
    return paginas_procesadas


def limpia_paginas(paginas_procesadas):
    paginas_limpias = []
    for linea in paginas_procesadas:
        if linea == 'Página':
            continue
        else:
            paginas_limpias.append(linea)
    return paginas_limpias


def elimina_tildes(s):
    replacements = (
        ('á', 'a'),
        ('é', 'e'),
        ('í', 'i'),
        ('ó', 'o'),
        ('ú', 'u'),
        ('à', 'a'),
        ('è', 'e'),
        ('ì', 'i'),
        ('ò', 'o'),
        ('ù', 'u'),
        ('ü', 'u'),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s


def transforma_hablante(linea):
    return elimina_tildes(linea.strip() \
                          .replace(' ', '') \
                          .replace('-', '') \
                          .replace(r'\x', '') \
                          .replace(',', '') \
                          ).lower()


def extrae_texto(paginas_limpias, documento):
    sw_sumario_encontrado = False
    sw_linea_inicio = False
    sw_inicio_encontrado = False
    linea_inicio = ''
    texto = []
    # Buscamos el inicio que está en la siguiente línea despues de SUMARIO
    for linea in paginas_limpias:
        if sw_sumario_encontrado == False:
            if linea == 'SUMARIO':
                sw_sumario_encontrado = True
        else:
            if sw_linea_inicio == False:
                try:
                    linea_inicio = correccion_inicio[documento]
                except:
                    linea_inicio = linea

                sw_linea_inicio = True
            else:
                if sw_inicio_encontrado == False:
                    if linea == linea_inicio:
                        sw_inicio_encontrado = True
                else:
                    # eliminamos las que contienen texto en mayusculas
                    if not linea[:5].isupper():
                        texto.append(linea)
    return texto


def agrupa_dialogos(texto):
    # limpieza de texto
    # Agrupamos discursos, un discurso empieza con el texto 'La señora' o 'El señor', asumimos que lo que no comience
    # por esto, pertenecerá al anterior hablante detectado
    dialogos_agrupados = []
    dialogo = ''
    for linea in texto:
        if (':' in linea and (linea.startswith('La señora') or linea.startswith('El señor'))) and \
                (linea[:50] not in excepciones_texto):
            dialogos_agrupados.append(" ".join(dialogo.split()))
            dialogo = ''
        dialogo = dialogo + ' ' + linea
    return dialogos_agrupados


def remove_punctuation(text):
    simbolos = '¿?¡!-0123456789‘’―—─‒‹›<>°´ºª'
    output = re.sub(r'\(.*?\)', '', text)
    output = re.sub('[%s]' % re.escape(string.punctuation), '', output)
    output = re.sub('[%s]' % re.escape(simbolos), '', output)
    return output


def devuelve_fecha(cadena):
    meses = {
        'enero': '01',
        'febrero': '02',
        'marzo': '03',
        'abril': '04',
        'mayo': '05',
        'junio': '06',
        'julio': '07',
        'agosto': '08',
        'septiembre': '09',
        'octubre': '10',
        'noviembre': '11',
        'diciembre': '12'
    }
    temp = cadena.split(' ')

    fecha = temp[7] + '-' + meses[temp[5].lower()] + '-' + temp[3].rjust(2, '0')
    valida = datetime.strptime(fecha, '%Y-%m-%d')
    return fecha


def elimina_puntuacion(dialogos_agrupados, fecha_proceso):
    dialogos = []
    for linea in dialogos_agrupados:
        ix = linea.find(':')

        if ix != -1:
            if transforma_hablante(linea[:ix]) not in excluidos:
                dialogos.append({'hablante': linea[:ix].strip(),
                                 'hablante_transformado': transforma_hablante(linea[:ix]),
                                 'discurso': linea[ix + 1:],
                                 'fecha_proceso': devuelve_fecha(fecha_proceso)})
    return dialogos


def proceso(documento):
    global contador_documentos
    diálogos_procesados = []
    reader = PdfReader(documento)
    paginas, fecha_proceso = extrae_paginas(reader)
    paginas_procesadas = procesa_paginas(paginas, documento)
    paginas_limpias = limpia_paginas(paginas_procesadas)
    texto = extrae_texto(paginas_limpias, documento)
    dialogos_agrupados = agrupa_dialogos(texto)
    dialogos = elimina_puntuacion(dialogos_agrupados, fecha_proceso)

    if len(dialogos) == 0 and documento not in documentos_sin_dialogos:
        print(documento)
        print('dialogos')
        print(dialogos)
        a = 0
        b = 7 / a

    hablantes_faltantes = []
    hablantes = []
    for i in dialogos:
        hablantes.append(i['hablante_transformado'])
    for el in list(set(hablantes)):
        try:
            a = dict_hablantes[el]
        except:
            hablantes_faltantes.append(el)

    if len(hablantes_faltantes) > 0 and documento not in documentos_sin_dialogos:
        print(documento)
        print('hablantes')
        print(hablantes_faltantes)
        a = 0
        b = 7 / a

    if documento not in documentos_sin_dialogos:
        for linea in dialogos:
            diálogos_procesados.append({'hablante': dict_hablantes[linea['hablante_transformado']],
                                        'discurso': linea['discurso'],
                                        'fecha_proceso': linea['fecha_proceso']
                                        })
    contador_documentos = contador_documentos + 1

    if (contador_documentos % 10) == 0:
        print(str(datetime.now()) + ' ' + str(contador_documentos))
    return diálogos_procesados


def lemmatizar(parametros):
    global contador
    documento = parametros[0]
    num_discurso = parametros[1]
    nlp = spacy.load('es_dep_news_trf')
    doc = nlp(remove_punctuation(documento['discurso'].strip()))
    lemmas = [[tok.lemma_,
               tok.orth_,
               documento['hablante']['nombre'],
               documento['hablante']['grupo parlamentario'],
               documento['fecha_proceso'],
               documento['szigriszt_pazos'],
               documento['flesch_reading_ease'],
               tok.pos_,
               num_discurso,
               i
               ] for i, tok in enumerate(doc) if (tok.pos_ not in ['PRON', 'ADP', 'DET', 'SCONJ', 'CCONJ', 'SPACE']) &
              (not tok.is_punct | tok.is_stop)]

    contador = contador + 1

    if (contador % 100) == 0:
        print(str(datetime.now()) + ' ' + str(contador))

    return pd.DataFrame(lemmas,
                        columns=['lemma', 'palabra', 'hablante', 'grupo parlamentario', 'fecha', 'szigriszt_pazos',
                                 'flesch_reading_ease', 'tipo palabra', 'numero_discurso', 'orden'])


def devuelve_polaridad(row):
    # como no me fio de los modelos calculamos la polaridad en 2 de ellos y calculamos la media
    translator = Translator(from_lang='es', to_lang='en')
    sentiment = SentimentIntensityAnalyzer()
    try:
        # la polaridad va entre -1 y +1
        translation = translator.translate(row['lemma'].strip().lower()).strip().lower()
        return (TextBlob(translation).sentiment.polarity + sentiment.polarity_scores(translation)['compound']) / 2
    except:
        return 0.0


def insertar_hablante(row):
    # Conexion a la base de datos Neo4j
    graph = row['graph']

    # Insertar nodo
    return graph.run("CREATE (h:Hablante {nombre: $name, grupo_parlamentario: $grupo})",
                     name=row['hablante'], grupo=row['grupo parlamentario']).stats()['nodes_created']


def insertar_lemma(row):
    # Conexion a la base de datos Neo4j
    graph = row['graph']

    # Insertar nodo
    return graph.run(
        "CREATE (p:Palabra {lemma: $lemma, palabras_agrupadas: $grupo, tipos_palabra: $tipos, \
        polaridad: $pol, polaridad_round: $pol_round})",
        lemma=row['lemma'],
        grupo=row['palabras agrupadas unicas'],
        tipos=row['tipo palabra agg'],
        pol=row['polaridad'],
        pol_round=row['polaridad_redondeada']
    ).stats()['nodes_created']


def obtener_metrica_szigriszt_pazos(discurso):
    '''
    P	estilo	tipo de publicación	estudios
    0 a 15	muy difícil	científica, filosófica	titulados universitarios
    16 a 35	árido	pedagógica, técnica	selectividad y estudios universitarios
    36 a 50	bastante difícil	literatura y divulgación	cursos secundarios
    51 a 65	normal	Los media	popular
    66 a 75	bastante fácil	novela, revista	12 años
    76 a 85	fácil	para quioscos	11 años
    86 a 100	muy fácil	cómics, tebeos y viñetas	6 a 10 años
    '''
    return textstat.szigriszt_pazos(re.sub(r'\(.*?\)', '', discurso).strip())


def obtener_metrica_flesch_reading_ease(discurso):
    '''
    Score 	Difficulty
    90-100 	Very Easy
    80-89 	Easy
    70-79 	Fairly Easy
    60-69 	Standard
    50-59 	Fairly Difficult
    30-49 	Difficult
    0-29 	Very Confusing
    '''
    return textstat.flesch_reading_ease(re.sub(r'\(.*?\)', '', discurso).strip())


def generar_dialogos(dialogos):
    if os.path.exists(dialogos):
        with open(dialogos, "rb") as f:
            dialogos_completos = pickle.load(f)
    else:
        lista_documentos = [arch for arch in ls() if arch.endswith('.PDF')]

        print(str(datetime.now()) + ' ' + 'Numero de documentos: ' + str(len(lista_documentos)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = [executor.submit(proceso, doc) for doc in lista_documentos]
            concurrent.futures.wait(results)
            dialogos_completos = []
            for f in results:
                dialogos_completos.extend(f.result())

        print(str(datetime.now()) + ' ' + 'Inicio métricas de legibilidad')
        # Calculo de metricas de legibilidad:
        dialogos_completos_metricas = []
        for elem in dialogos_completos:
            elem['szigriszt_pazos'] = obtener_metrica_szigriszt_pazos(elem['discurso'])
            elem['flesch_reading_ease'] = obtener_metrica_flesch_reading_ease(elem['discurso'])
            dialogos_completos_metricas.append(elem)

        dialogos_completos = dialogos_completos_metricas
        print(str(datetime.now()) + ' ' + 'Fin métricas de legibilidad')

        with open(dialogos, "wb") as f:
            pickle.dump(dialogos_completos, f)

    print(str(datetime.now()) + ' ' + 'Diálogos Generados')
    return dialogos_completos


def generar_lemmas(palabras_lemm, dialogos_completos):
    print(str(datetime.now()))
    if os.path.exists(palabras_lemm):
        print(str(datetime.now()) + ' ' + 'Cargando Lemmas')
        with open(palabras_lemm, "rb") as f:
            df_lemmas = pickle.load(f)
        print(str(datetime.now()) + ' ' + 'Lemmas cargados')
    else:

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = [executor.submit(lemmatizar, [elem, num]) for num, elem in enumerate(dialogos_completos)]
            concurrent.futures.wait(results)
            df_lemmas = pd.DataFrame()
            print(str(datetime.now()) + ' ' + 'procesando resultados')
            for f in results:
                df_lemmas = pd.concat([df_lemmas, f.result()], axis=0)
            df_lemmas['lemma'] = df_lemmas['lemma'].str.lower()
            # Eiminamos los espacios sueltos
            df_lemmas['lemma'].fillna(' ', inplace=True)
            df_lemmas = df_lemmas[df_lemmas['lemma'] != ' ']

        print(str(datetime.now()) + ' ' + 'salvando lemmas')
        with open(palabras_lemm, "wb") as f:
            pickle.dump(df_lemmas, f)
        print(str(datetime.now()) + ' ' + 'lemmas salvados')

    print(str(datetime.now()) + ' ' + 'Lemmas Generados')
    return df_lemmas


def imprimir_resultados(dialogos_completos):
    # Imprimir resultados
    print('numero de intervenciones: ' + str(len(dialogos_completos)))
    palabras = []
    for el in dialogos_completos:
        palabras.extend(el['discurso'].split(' '))
    print('número de plabras: ' + str(len(palabras)))


def generar_df_agrupado(df_lemmas):
    print('Total Lemmas: ' + str(len(df_lemmas)))
    df_agrupado = df_lemmas.groupby(['lemma', 'hablante', 'grupo parlamentario', 'fecha', 'tipo palabra'])[
        'palabra'].apply(list).reset_index()
    df_agrupado['num_palabras'] = df_agrupado['palabra'].apply(len)
    df_agrupado['palabras agrupadas'] = df_agrupado['palabra'].apply(lambda x: list(dict.fromkeys(x)))
    # Seleccionamos las 20 palabras más dichas para elminarlas
    df_grouped = df_agrupado.groupby(['lemma'])['num_palabras'].sum().reset_index()
    df_grouped.sort_values(by='num_palabras', ascending=False, inplace=True)
    top_20_lemmas = df_grouped.head(20)
    df_agrupado = df_agrupado[~df_agrupado['lemma'].isin(top_20_lemmas['lemma'])]
    # elminiamos las palabras que se dicén 3 veces o menos
    palabras_menos_dichas = df_grouped[df_grouped['num_palabras'] <= 3]
    df_agrupado = df_agrupado[~df_agrupado['lemma'].isin(palabras_menos_dichas['lemma'])]
    print('20 Lemmas más repetidos:')
    print(top_20_lemmas)
    print(str(datetime.now()) + ' ' + 'registros df_agrupado: ' + str(len(df_agrupado)))

    # descartamos los valores atípicos
    # voy a cancelar esto, para poner todas las palabras
    '''
    df_sin_atipicos = df_agrupado[['hablante', 'lemma', 'num_palabras', 'fecha']].groupby(['lemma']).agg(
        {'num_palabras': sum}).reset_index()
    avg = df_sin_atipicos['num_palabras'].mean()
    stDev = df_sin_atipicos['num_palabras'].std()
    print('Media: ' + str(avg))
    print('stDev: ' + str(stDev))
    # descartamos todas las palabras que se hayan dicho menos de 5 veces
    df_sin_atipicos = df_sin_atipicos[df_sin_atipicos['num_palabras'] >= 3]
    limite_maximo = round(avg + (stDev * 1.5))
    # descartamos todas las palabras que se hayan dicho más veces que la media + 1*5 la desviación típica
    df_sin_atipicos = df_sin_atipicos[df_sin_atipicos['num_palabras'] < limite_maximo]
    df_sin_atipicos = df_sin_atipicos.drop(columns=['num_palabras'])
    print(str(datetime.now()) + ' ' + 'registros sin atípicos: ' + str(len(df_sin_atipicos)))

    df_agrupado = pd.merge(df_agrupado, df_sin_atipicos, on='lemma')
    print(str(datetime.now()) + ' ' + 'Nueva longitud df_agrupado: ' + str(len(df_sin_atipicos)))
    '''
    return df_agrupado, top_20_lemmas, palabras_menos_dichas


def generar_lemmas_unicos(df_agrupado):
    df_lemmas_unicos = df_agrupado.groupby(['lemma']).agg(
        {'palabras agrupadas': sum, 'num_palabras': sum, 'tipo palabra': list}).reset_index()
    df_lemmas_unicos['palabras agrupadas unicas'] = df_lemmas_unicos['palabras agrupadas'].apply(
        lambda x: list(dict.fromkeys(x)))
    df_lemmas_unicos['tipo palabra agg'] = df_lemmas_unicos['tipo palabra'].apply(lambda x: list(dict.fromkeys(x)))
    df_lemmas_unicos = df_lemmas_unicos.drop('palabras agrupadas', axis=1)
    df_lemmas_unicos.sort_values(by=['num_palabras'], inplace=True, ascending=False)
    df_lemmas_unicos = df_lemmas_unicos[['lemma', 'num_palabras', 'palabras agrupadas unicas', 'tipo palabra agg']]
    print(str(datetime.now()) + ' ' + 'polaridad inicio')
    # df_lemmas_unicos['polaridad'] = df_lemmas_unicos.apply(devuelve_polaridad, axis=1)

    # crear un DataFrame Dask a partir del DataFrame de Pandas
    meta = pd.Series([], dtype='float64')
    df_dask = dd.from_pandas(df_lemmas_unicos, npartitions=10)

    # aplicar la función devuelve_polaridad en paralelo a cada partición del DataFrame
    df_dask['polaridad'] = df_dask.map_partitions(lambda df: df.apply(devuelve_polaridad, axis=1), meta=meta)

    # convertir el DataFrame Dask de vuelta a un DataFrame de Pandas
    df_lemmas_unicos = df_dask.compute()

    print(str(datetime.now()) + ' ' + 'polaridad fin')
    df_lemmas_unicos['polaridad_redondeada'] = df_lemmas_unicos['polaridad'].apply(lambda x: round(x, 1))
    return df_lemmas_unicos


def insertar_lemma_Neo4j(graph, df_lemmas_unicos):
    df_resultados_lemmas = df_lemmas_unicos[['lemma', 'num_palabras', 'palabras agrupadas unicas',
                                             'tipo palabra agg', 'polaridad', 'polaridad_redondeada']]
    df_resultados_lemmas['graph'] = graph

    print('Lemmas únicos: ' + str(len(df_resultados_lemmas)))
    print(str(datetime.now()) + ' ' + 'Inserta Lemmas')
    results = []
    for i in range(0, len(df_resultados_lemmas), 10):
        if (i % 10000) == 0:
            print(str(datetime.now()) + ' ' + str(i))
        block = df_resultados_lemmas[i:i + 10]
        results.extend(block.apply(insertar_lemma, axis=1).tolist())
    df_lemmas_unicos['resultado'] = results
    print(str(datetime.now()) + ' ' + 'Fin Inserta Lemmas')
    print(df_lemmas_unicos[df_lemmas_unicos['resultado'] != 1].to_string())


def generar_hablantes_unicos(df_agrupado):
    df_hablantes_unicos = df_agrupado.drop_duplicates(['hablante', 'grupo parlamentario'])
    df_hablantes_unicos = df_hablantes_unicos[['hablante', 'grupo parlamentario']]
    return df_hablantes_unicos


def insertar_hablantes_Neo4j(graph, df_hablantes_unicos):
    df_resultados_hablantes = df_hablantes_unicos[['hablante', 'grupo parlamentario']]
    df_resultados_hablantes['graph'] = graph
    print(str(datetime.now()) + ' ' + 'Inserta Hablantes')
    df_resultados_hablantes['resultado'] = df_resultados_hablantes.apply(insertar_hablante, axis=1)
    print(str(datetime.now()) + ' ' + 'Fin Inserta Hablantes')
    print(df_resultados_hablantes[df_resultados_hablantes['resultado'] != 1].to_string())


def insertar_relacion_dice(row):
    # Conexion a la base de datos Neo4j
    graph = row['graph']

    # Insertar nodo
    return graph.run(
        'MATCH (n:Hablante) \
         MATCH (p:Palabra) \
         WHERE n.nombre =  $nombre \
         AND p.lemma = $lemma \
         CREATE (n)-[r:DICE {fecha: $fecha, veces: $veces, grupo_parlamentario: $grupo}]->(p) \
         RETURN COUNT(r)',
        nombre=row['hablante'],
        lemma=row['lemma'],
        fecha=row['fecha'],
        veces=row['num_palabras'],
        grupo=row['grupo parlamentario']
    ).stats()['relationships_created']


def insertar_relacion_DICE_Neo4j(graph, df_agrupado):
    df_relaciones = df_agrupado.groupby(['hablante', 'grupo parlamentario', 'fecha', 'lemma'])[
        'num_palabras'].sum().reset_index()
    df_relaciones['graph'] = graph
    print('Relaciones totales: ' + str(len(df_relaciones)))
    print(str(datetime.now()) + ' ' + 'Inserta Relacion DICE')
    results = []
    for i in range(0, len(df_relaciones), 10):
        if i % 1000000 == 0:
            print(str(datetime.now()) + ' ' + str(i))
        block = df_relaciones[i:i + 10]
        results.extend(block.apply(insertar_relacion_dice, axis=1).tolist())
    df_relaciones['resultado'] = results
    print(str(datetime.now()) + ' ' + 'FIN Inserta Relacion DICE')
    print(df_relaciones[df_relaciones['resultado'] != 1].to_string())


def insertar_relacion_discurso(row):
    # Conexion a la base de datos Neo4j
    graph = row['graph']
    # Insertar nodo
    return graph.run(
        'MATCH (p1:Palabra) \
         MATCH (p2:Palabra) \
         WHERE p1.lemma = $lemma_orig \
         AND p2.lemma = $lemma_dest \
         CREATE (p1)-[r:DISCURSO {grupo_parlamentario: $grupo, hablante:$hablante, \
         fecha:$fecha, num_discruso:$num_discurso, orden:$orden}]->(p2) \
         RETURN COUNT(r)',
        lemma_orig=row['lemma'],
        lemma_dest=row['lemma_siguiente'],
        grupo=row['grupo parlamentario'],
        hablante=row['hablante'],
        fecha=row['fecha'],
        num_discurso=row['numero_discurso'],
        orden=row['orden'],
    ).stats()['relationships_created']


def insertar_relacion_DISCURSO_Neo4j(graph, df_lemmas, top_20_lemmas, palabras_menos_dichas):
    # eliminamos las 20 palabras más dichas
    df_lemmas = df_lemmas[~df_lemmas['lemma'].isin(top_20_lemmas['lemma'])]
    # eliminamos las palabras que se dicen 3 veces o menos
    df_lemmas = df_lemmas[~df_lemmas['lemma'].isin(palabras_menos_dichas['lemma'])]

    df_lemmas.sort_values(by=['hablante', 'grupo parlamentario', 'fecha', 'numero_discurso', 'orden', 'lemma'],
                          inplace=True, ascending=True)
    df_ordenado = df_lemmas.groupby(['hablante', 'grupo parlamentario', 'fecha', 'numero_discurso'], group_keys=False)
    df_ordenado = df_ordenado.apply(lambda x: x.assign(lemma_siguiente=x['lemma'].shift(-1)))
    df_discursos = df_ordenado[
        ['hablante', 'grupo parlamentario', 'fecha', 'numero_discurso', 'lemma', 'lemma_siguiente', 'orden']]
    df_discursos = df_discursos[~df_discursos['lemma_siguiente'].isnull()]
    df_discursos['graph'] = graph

    print('Numero de Relaciones DISCURSO: ' + str(len(df_discursos)))
    print(str(datetime.now()) + ' ' + 'INICIO Inserta Relacion DISCURSO')
    results = []
    for i in range(0, len(df_discursos), 10):
        if i % 1000000 == 0:
            print(str(datetime.now()) + ' ' + str(i))
        block = df_discursos[i:i + 10]
        results.extend(block.apply(insertar_relacion_discurso, axis=1).tolist())
    df_discursos['resultado'] = results
    print(str(datetime.now()) + ' ' + 'FIN Inserta Relacion DISCURSO')
    print(df_discursos[df_discursos['resultado'] != 1].to_string())


def main():
    dialogos = 'dialogos.pickle'
    palabras_lemm = 'palabras.pickle'
    textstat.set_lang('es')

    graph = conecta_neo4j()

    dialogos_completos = generar_dialogos(dialogos)

    imprimir_resultados(dialogos_completos)

    df_lemmas = generar_lemmas(palabras_lemm, dialogos_completos)

    # Esto es una chapuza y si puedo reescribiré el código
    df_agrupado, top_20_lemmas, palabras_menos_dichas = generar_df_agrupado(df_lemmas)

    df_lemmas_unicos = generar_lemmas_unicos(df_agrupado)

    insertar_lemma_Neo4j(graph, df_lemmas_unicos)

    df_hablantes_unicos = generar_hablantes_unicos(df_agrupado)

    insertar_hablantes_Neo4j(graph, df_hablantes_unicos)

    print('Lemmas únicos: ' + str(len(df_lemmas_unicos)))
    print('Hablantes únicos: ' + str(len(df_hablantes_unicos)))

    insertar_relacion_DICE_Neo4j(graph, df_agrupado)

    insertar_relacion_DISCURSO_Neo4j(graph, df_lemmas, top_20_lemmas, palabras_menos_dichas)

    print(str(datetime.now()) + ' ' + 'PROCESO TERMINADO')


if __name__ == '__main__':
    global contador
    contador = 0
    global contador_documentos
    contador_documentos = 0

    main()

# https://rpubs.com/Andres25/791004


'''
CALL gds.graph.project(
  'pageRank_DICE',
  ['Hablante', 'Palabra'],
  'DICE',
  {
    relationshipProperties: 'veces'
  }
)

CALL gds.pageRank.write('pageRank_DICE', {
  maxIterations: 20,
  dampingFactor: 0.85,
  writeProperty: 'pagerank_DICE',
  relationshipWeightProperty: 'veces'
})
YIELD nodePropertiesWritten, ranIterations

CALL gds.articleRank.write('pageRank_DICE', {
  writeProperty: 'centrality_DICE',
  relationshipWeightProperty: 'veces'
})
YIELD nodePropertiesWritten, ranIterations

match (h:Hablante)-[d:DICE]->(p:Palabra)
WITH p, COUNT(d) as num
WHERE num = 1
RETURN COUNT(p), max(p.pagerank_DICE)

CALL gds.graph.project(
  'pageRank_DISCURSO',
  ['Palabra'],
  'DISCURSO'
)

CALL gds.pageRank.write('pageRank_DISCURSO', {
  maxIterations: 20,
  dampingFactor: 0.85,
  writeProperty: 'pagerank_DISCURSO'
})
YIELD nodePropertiesWritten, ranIterations

CALL gds.louvain.write('pageRank_DISCURSO', {
  writeProperty: 'Communities',
  includeIntermediateCommunities: true,
  maxIterations: 1000
})
YIELD communityCount, modularity, modularities


CALL gds.graph.project.cypher(
  'discurso_VOX',
  'MATCH (p:Palabra) RETURN id(p) AS id',
  'MATCH (p:Palabra)-[r:DISCURSO]->(p2:Palabra) WHERE r.grupo_parlamentario = "Grupo Parlamentario VOX" RETURN id(p) AS source, id(p2) AS target')
YIELD
  graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels
  

CALL gds.louvain.write('discurso_VOX', {
  writeProperty: 'Communities_VOX',
  includeIntermediateCommunities: true,
  maxIterations: 1000
})
YIELD communityCount, modularity, modularities


MATCH (p:Palabra)
UNWIND p.Communities AS comunidad
RETURN p.lemma, comunidad
limit 10

MATCH (p:Palabra)
UNWIND p.Communities_2023_02_09 AS comunidad
WITH p.lemma as palabra, comunidad
return comunidad, count(palabra) as num 
order by num desc
limit 10


-------estudios por fechas
  
CALL gds.graph.project.cypher(
  'DICE_2023-02-09',
  'MATCH (n) RETURN id(n) AS id, labels(n) as labels',
  'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) WHERE d.fecha = "2023-02-09" RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces')
YIELD
  graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels


CALL gds.pageRank.write('DICE_2023-02-09', {
  maxIterations: 20,
  dampingFactor: 0.85,
  writeProperty: 'pagerank_DICE_2023_02_09',
  relationshipWeightProperty: 'veces'
})
YIELD nodePropertiesWritten, ranIterations

CALL gds.louvain.write('DICE_2023-02-09', {
  writeProperty: 'Communities_2023_02_09',
  includeIntermediateCommunities: false,
  maxIterations: 1000
})
YIELD communityCount, modularity, modularities

CALL gds.graph.project.cypher(
  'DISCURSO_2023-02-09_VOX',
  'MATCH (n) RETURN id(n) AS id, labels(n) as labels',
  'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) WHERE d.fecha = "2023-02-09" AND d.grupo_parlamentario = "Grupo Parlamentario VOX" RETURN id(p1) AS source, id(p2) AS target, type(d) AS type')
YIELD
  graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels
  
CALL gds.graph.project.cypher(
  'DICE_2023-02-09_VOX',
  'MATCH (n) RETURN id(n) AS id, labels(n) as labels',
  'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) WHERE d.fecha = "2023-02-09" AND d.grupo_parlamentario = "Grupo Parlamentario VOX" RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces')
YIELD
  graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels
  
CALL gds.pageRank.write('DICE_2023-02-09_VOX', {
  maxIterations: 20,
  dampingFactor: 0.85,
  writeProperty: 'pagerank_DICE_2023_02_09_VOX',
  relationshipWeightProperty: 'veces'
})
YIELD nodePropertiesWritten, ranIterations

CALL gds.louvain.write('DISCURSO_2023-02-09_VOX', {
  writeProperty: 'Communities_2023_02_09_VOX',
  includeIntermediateCommunities: false,
  maxIterations: 1000
})
YIELD communityCount, modularity, modularities


CALL gds.graph.project.cypher(
  'DICE_2023-02-09_PODEMOS',
  'MATCH (n) RETURN id(n) AS id, labels(n) as labels',
  'MATCH (h:Hablante)-[d:DICE]->(p:Palabra) WHERE d.fecha = "2023-02-09" AND d.grupo_parlamentario = "Grupo Parlamentario Confederal de Unidas Podemos-En Comú Podem-Galicia en Común" RETURN id(h) AS source, id(p) AS target, type(d) AS type, d.veces as veces')
YIELD
  graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels
  
  
CALL gds.graph.project.cypher(
  'DISCURSO_2023-02-09_PODEMOS',
  'MATCH (n) RETURN id(n) AS id, labels(n) as labels',
  'MATCH (p1:Palabra)-[d:DISCURSO]->(p2:Palabra) WHERE d.fecha = "2023-02-09" AND d.grupo_parlamentario = "Grupo Parlamentario Confederal de Unidas Podemos-En Comú Podem-Galicia en Común" RETURN id(p1) AS source, id(p2) AS target, type(d) AS type')
YIELD
  graphName AS graph, nodeQuery, nodeCount AS nodes, relationshipQuery, relationshipCount AS rels
  
  
CALL gds.pageRank.write('DICE_2023-02-09_PODEMOS', {
  maxIterations: 20,
  dampingFactor: 0.85,
  writeProperty: 'pagerank_DICE_2023_02_09_PODEMOS',
  relationshipWeightProperty: 'veces'
})
YIELD nodePropertiesWritten, ranIterations

CALL gds.louvain.write('DISCURSO_2023-02-09_PODEMOS', {
  writeProperty: 'Communities_2023_02_09_PODEMOS',
  includeIntermediateCommunities: false,
  maxIterations: 1000
})
YIELD communityCount, modularity, modularities




MATCH (p:Palabra) WHERE p.lemma = 'animal' 
WITH p.Communities_2023_02_09_PODEMOS AS comunidad
MATCH (p:Palabra)
WHERE p.Communities_2023_02_09_PODEMOS = comunidad
RETURN p.lemma, comunidad



DESVIACION TIPICA
MATCH (h:Hablante)-[d:DICE]->(p:Palabra)
WITH  p.lemma as palabra, sum(d.veces) as veces
WITH collect(veces) AS veces_values, palabra, veces
UNWIND veces_values as valores
WITH collect(valores) as valores_stDEv
RETURN apoc.coll.stdev(valores_stDEv) 

Promedio ±1,5 desviación estándar

https://help.highbond.com/helpdocs/analytics/141/user-guide/es/Content/analyzing_data/identifying_outliers.html


'''
