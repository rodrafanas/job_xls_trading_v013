import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import openpyxl
import io
import json

from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

warnings.filterwarnings("ignore")


def extract_table_from_file(path_to_file):
    # st.write(path_to_file.name)
    if path_to_file != 'resumo.xlsx':
        df = pd.read_excel(path_to_file)
        cols = df.iloc[2].to_list()
        cols = ['NaN' if pd.isna(valor) else valor for valor in cols]
        
        df = df.set_axis(cols, axis = 1)
        df = df.iloc[5:]
        indice_final = df[df["Mic"].isna()].index[0]
        df =  df.drop(df.index[indice_final-5:])
        lote = path_to_file.name.replace('.xlsx', '')
        df['Lote'] = lote
        # lote = df['Lote'].str.replace('/', '', regex=False)
        # df['Lote'] = lote   
        df['COR'] = '31-4'

        # Verifique se a coluna 'LEAF' existe no DataFrame
        if 'LEAF' not in df.columns:
            # Se não existir, crie uma coluna 'LEAF' com valores vazios
            df['LEAF'] = np.nan

        sel_cols = ['Lote','Fardo','P. Líquido', 'Mic', 'UHM', 'Res', 'COR', 'LEAF']
        df = df[sel_cols]
    else:
        df = pd.read_excel(path_to_file)

    return df

def run_extract_table(files):
    # Itere sobre cada arquivo
    df = pd.DataFrame()
    for file in files:    
        table = extract_table_from_file(file)
        if file.name != 'resumo.xlsx':
            df =  pd.concat([df,table])
        else:
            df = table

    return df

def gera_df(files):
    # Extração dos arquivos xlsx
    df = run_extract_table(files)
    return df

def carrega_parms(parms_file):
    file_parms = parms_file['parms.json']
    params = json.load(file_parms)      
    return params



def selecionar_lotes():
    uploaded_files = st.file_uploader("Faça upload dos arquivos: ", accept_multiple_files=True, type=["xlsx","json"])

    if uploaded_files != {}:
        # Dicionários para separar os arquivos
        xlsx_files = {}
        resumo_file = {}
        parms_file = {}


        for idx, uploaded_file in enumerate(uploaded_files):
            # Verifica se é o arquivo parms.json
            if uploaded_file.name == 'parms.json':
                parms_file[uploaded_file.name] = uploaded_file
            # Verifica se é o arquivo resumo.xlsx
            elif uploaded_file.name == 'resumo.xlsx':
                resumo_file[uploaded_file.name] = uploaded_file
            # Todos os outros arquivos .xlsx
            elif uploaded_file.name.endswith('.xlsx') and uploaded_file.name != 'resumo.xlsx':
                xlsx_files[int(idx)] = uploaded_file
    return xlsx_files, resumo_file, parms_file

st.title("Editor de Arquivos Excel")

xlsx_files, resumo_file, parms_file = selecionar_lotes()

## xlsx
if xlsx_files != {}:
    st.header('xlsx files')
    xlsx_files_mod = list(xlsx_files.values())
    dfx = gera_df(xlsx_files_mod)
    st.data_editor(dfx, key = 'dfx')



# Função auxiliar para converter DataFrame para Excel
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        # writer.save()
    processed_data = output.getvalue()
    return processed_data



## resumo
if resumo_file != {}:
    st.header("resumo.xlsx file")
    df_resumo = pd.read_excel(resumo_file["resumo.xlsx"])
    st.data_editor(df_resumo, key= 'df_resumo')
    # df_resumo_xlsx = to_excel(df_resumo)
    # Botão de download
    st.download_button(label='Download Excel',
                    data=to_excel(df_resumo),
                    file_name='resumo_output.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


    # st.download_button(label='Download resumo.xlsx',
    #                             data=df_resumo_xlsx ,
    #                             file_name= 'df_resumo_test.xlsx')


    # st.download_button(label="Download resumo.xlsx", data=df_resumo, file_name="resumo_test.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


## parms 
if parms_file != {}:
    st.header('parms.json file')
    params = carrega_parms(parms_file)
    st.write(params)
    
