import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import io
import json

warnings.filterwarnings("ignore")

def extract_table_from_file(path_to_file):
    # st.write(path_to_file.name)
    if path_to_file.name != 'resumo.xlsx':
        df = pd.read_excel(path_to_file)

        indice_inicio = df.head(7).T.isna().sum().index[df.head(7).T.isna().sum()<4][0]
        cols = df.iloc[indice_inicio].to_list()
        cols = ['NaN' if pd.isna(valor) else valor for valor in cols]
        # print(cols)
        df = df.set_axis(cols, axis = 1)
        drop_indices = df.head(7).T.isna().sum().index[df.head(7).T.isna().sum()>4].tolist()
        # Removendo as linhas com base nos índices
        df = df.drop(drop_indices).iloc[1:] #, inplace=True)
        df = df.reset_index(drop=True)
        indice_final = df[df["Mic"].isna()].index[0]
        df =  df.drop(df.index[indice_final:])
        lote = path_to_file.name.replace('.xlsx', '')
        df['Lote'] = lote
        # lote = df['Lote'].str.replace('/', '', regex=False)
        # df['Lote'] = lote   
        # df['COR'] = '31-4'

        # Verifique se a coluna 'LEAF' existe no DataFrame
        if 'LEAF' not in df.columns:
            # Se não existir, crie uma coluna 'LEAF' com valores vazios
            df['LEAF'] = np.nan

        # Verifique se a coluna 'COR' existe no DataFrame
        if 'COR' not in df.columns:
            # Se não existir, crie uma coluna 'COR' com valores vazios
            df['COR'] = np.nan

        sel_cols = ['Lote','Fardo','P. Líquido', 'Mic', 'UHM', 'Res', 'COR', 'LEAF']
        df = df[sel_cols]
    else:
        df = pd.read_excel(path_to_file)

    return df

def run_extract_table(files):
    
    # Itere sobre cada arquivo
    df = pd.DataFrame()
    for file in files.values():    
        
        table = extract_table_from_file(file)
        if file.name != 'resumo.xlsx':
            df =  pd.concat([df,table])
        else:
            df = table

    return df


def stats_table(df,slider_bales_before=28, option_res= 'acima',
                # slider_mic_min=3.86, slider_mic_max=4.50, 
                slider_mic = (3.58,4.5),
                slider_uhm=1.10, option_uhm= 'acima'):
                # slider_mic=df.Mic.mean().round(2), option_mic= 'acima',
                # slider_uhm=df.UHM.mean().round(2), option_uhm= 'acima',):
    # tratamento da Cor
    df['COR'] = df['COR'].str.split('-').str[0]
    df['COR'] = df['COR'].fillna(0)
    df.dropna(inplace=True)
    # Agrupa por 'lote' e calcula a estatistica 
    df['UHM'] = df['UHM'].astype(float)
    resultados = df.groupby('Lote').agg(P_Liq_sum=pd.NamedAgg(column='P. Líquido', aggfunc=np.sum),
                                        Mic_avg=pd.NamedAgg(column='Mic', aggfunc=np.mean),
                                        Mic_min=pd.NamedAgg(column='Mic', aggfunc=np.min),
                                        Mic_max=pd.NamedAgg(column='Mic', aggfunc=np.max),
                                        UHM_avg=pd.NamedAgg(column='UHM', aggfunc=np.mean),
                                        UHM_min=pd.NamedAgg(column='UHM', aggfunc=np.min),
                                        UHM_max=pd.NamedAgg(column='UHM', aggfunc=np.max),
                                        GPT_avg=pd.NamedAgg(column='Res', aggfunc=np.mean),
                                        GPT_min=pd.NamedAgg(column='Res', aggfunc=np.min),
                                        GPT_max=pd.NamedAgg(column='Res', aggfunc=np.max),
                                        GPT_90=pd.NamedAgg(column='Res', aggfunc=lambda x: np.percentile(x, q=90)),
                                        ).reset_index()
    ## Res
    if option_res == 'acima':
        Res = df.groupby('Lote').agg(Bales_below_28=pd.NamedAgg(column='Res', aggfunc=lambda x: np.count_nonzero(x>=slider_bales_before)/np.count_nonzero(x))).reset_index()
    elif option_res == 'abaixo':
        Res = df.groupby('Lote').agg(Bales_below_28=pd.NamedAgg(column='Res', aggfunc=lambda x: np.count_nonzero(x<=slider_bales_before)/np.count_nonzero(x))).reset_index()

    resultados = pd.merge(resultados,Res,how='left',on='Lote')

    ## LEAF
    LEAF_Avg = df.groupby('Lote')['LEAF'].mean().reset_index()

    resultados = pd.merge(resultados,LEAF_Avg,how='left',on='Lote')
   
    ## Mic
    Mic = df.groupby('Lote').agg(Mic_option=pd.NamedAgg(column='Mic', aggfunc=lambda x: np.count_nonzero((x>=slider_mic[0]) & (x<=slider_mic[1]))/np.count_nonzero(x)))

    # if option_mic == 'acima':
    #     Mic = df.groupby('Lote').agg(Mic_option=pd.NamedAgg(column='Mic', aggfunc=lambda x: np.count_nonzero(x>slider_mic)/np.count_nonzero(x))).reset_index()
    # elif option_mic == 'abaixo':
    #     Mic = df.groupby('Lote').agg(Mic_option=pd.NamedAgg(column='Mic', aggfunc=lambda x: np.count_nonzero(x<slider_mic)/np.count_nonzero(x))).reset_index()

    resultados = pd.merge(resultados,Mic,how='left',on='Lote')

    ## UHM
    if option_uhm == 'acima':
        UHM = df.groupby('Lote').agg(UHM_option=pd.NamedAgg(column='UHM', aggfunc=lambda x: np.count_nonzero(x>=slider_uhm)/np.count_nonzero(x))).reset_index()
    elif option_uhm == 'abaixo':
        UHM = df.groupby('Lote').agg(UHM_option=pd.NamedAgg(column='UHM', aggfunc=lambda x: np.count_nonzero(x<=slider_uhm)/np.count_nonzero(x))).reset_index()

    resultados = pd.merge(resultados,UHM,how='left',on='Lote')

    # Formatar a coluna como porcentagem   
    resultados['Bales_below_28'] = resultados['Bales_below_28'].mul(100).apply(lambda x: round(x, 1)) #.apply(lambda x: f'{x * 100:.2f}%') #.map("{:.0%}".format)
    resultados['Mic_option'] = resultados['Mic_option'].mul(100).apply(lambda x: round(x, 1)) #.apply(lambda x: f'{x * 100:.2f}%') #.map("{:.1%}".format)
    resultados['UHM_option'] = resultados['UHM_option'].mul(100).apply(lambda x: round(x, 1)) #.apply(lambda x: f'{x * 100:.2f}%') #.map("{:.1%}".format)
    resultados['Mic_avg'] = resultados['Mic_avg'].apply(lambda x: round(x, 1))
    resultados['Mic_min'] = resultados['Mic_min'].apply(lambda x: round(x, 1))
    resultados['Mic_max'] = resultados['Mic_max'].apply(lambda x: round(x, 1))
    resultados['UHM_avg'] = resultados['UHM_avg'].apply(lambda x: round(x, 2))
    resultados['UHM_min'] = resultados['UHM_min'].apply(lambda x: round(x, 2))
    resultados['UHM_max'] = resultados['UHM_max'].apply(lambda x: round(x, 2))
    resultados['GPT_avg'] = resultados['GPT_avg'].apply(lambda x: round(x, 1))
    resultados['GPT_min'] = resultados['GPT_min'].apply(lambda x: round(x, 1))
    resultados['GPT_max'] = resultados['GPT_max'].apply(lambda x: round(x, 1))
    resultados['GPT_90'] = resultados['GPT_90'].apply(lambda x: round(x, 1))
    resultados['LEAF'] = resultados['LEAF'].apply(lambda x: round(x, 2))

    ## traduzindo acima e abaixo para above and below
    if option_res == 'acima':
        option_res_trans = 'above'
    elif option_res == 'abaixo':
        option_res_trans = 'below'
    else:
        option_res_trans = option_res


    if option_uhm == 'acima':
        option_uhm_trans = 'above'
    elif option_uhm == 'abaixo':
        option_uhm_trans = 'below'
    else:
        option_uhm_trans = option_uhm

    # # Renomeia as colunas
    resultados.columns = ['Lote', 'Net weight', 'Mic (avg)', 'Mic (min)', 'Mic (max)',
                            'UHM Avg', 'UHM min', 'UHM max',
                            'GPT avg', 'GPT min', 'GPT Max', 'GPT 90%', 
                            f'GPT {option_res_trans} {slider_bales_before} (%)',
                            'LEAF Avg',
                            f'Mic between {slider_mic[0]} and {slider_mic[1]} (%)',
                            f'UHM {option_uhm_trans} {slider_uhm} (%)',]

    # def class_uhm(valor):
    #     if 0 >= valor <= 0.79:
    #         return 24
    #     elif 0.80 >= valor <= 0.85:
    #         return 26
    #     elif 0.86 >= valor <= 0.89:
    #         return 28
    #     elif 0.90 >= valor <= 0.92:
    #         return 29
    #     elif 0.93 >= valor <= 0.95:
    #         return 30
    #     elif 0.96 >= valor <= 0.98:
    #         return 31
    #     elif 0.99 >= valor <= 1.01:
    #         return 32
    #     elif 1.02 >= valor <= 1.04:
    #         return 33
    #     elif 1.05 >= valor <= 1.07:
    #         return 34
    #     elif 1.08 >= valor <= 1.10:
    #         return 35
    #     elif 1.11 >= valor <= 1.13:
    #         return 36
    #     elif 1.14 >= valor <= 1.17:
    #         return 37
    #     elif 1.18 >= valor <= 1.20:
    #         return 38
    #     elif 1.21 >= valor <= 1.23:
    #         return 39
    #     elif 1.24 >= valor <= 1.26:
    #         return 40
    #     elif 1.27 >= valor <= 1.29:
    #         return 41
    #     elif 1.30 >= valor <= 1.32:
    #         return 42
    #     elif 1.33 >= valor <= 1.35:
    #         return 43
    #     elif valor >= 1.36:
    #         return 44
    #     else:
    #         return np.nan  # Classificação padrão para valores fora das faixas

    def class_uhm(valor):
        bins = [0, 1.08, 1.11, 1.14, 1.18, 1.21, np.inf]
        labels = ['below_34', '35', '36', '37', '38', 'above_38']
        return pd.cut([valor], bins=bins, labels=labels, right=False)[0]


    # Aplicar a função à coluna 'valor' e criar uma nova coluna 'classificacao'
    df['UHM'] = df['UHM'].astype(float)
    df['UHM_class'] = df['UHM'].apply(class_uhm)

    UHM_class = pd.crosstab(df.Lote,df.UHM_class, margins=True, margins_name="total").add_prefix("Staples_").reset_index()

    UHM_class.rename(columns={"Staples_total":"Total_Bales"},inplace=True)

    cols_original = UHM_class.columns.tolist()

    cols_final = ['Lote', 'Staples_below_34', 'Staples_35','Staples_36', 'Staples_37','Staples_38','Staples_above_38', 'Total_Bales']


    cols_difference = list(set(cols_final) - set(cols_original))

    # Adicionar as colunas ausentes ao DataFrame, preenchidas com 0
    for col in cols_difference:
        UHM_class[col] = 0

    # Reordenar as colunas para corresponder à ordem em cols_final
    UHM_class = UHM_class[cols_final]

    resultados = pd.merge(resultados,UHM_class,how='left',on='Lote')

    COR_class = pd.crosstab(df.Lote,df.COR, margins=True, margins_name="total", normalize='index').add_prefix("COR_").reset_index()

    COR_class.rename(columns={"COR_total":"Total_COR"},inplace=True)
    
    cols = list(set(COR_class.columns.tolist()) - set(['Lote']))
    
    for col in cols:
        COR_class[col] = COR_class[col].mul(100).apply(lambda x: round(x, 2))

    resultados = pd.merge(resultados,COR_class,how='left',on='Lote')

    resultados['OFFERED'] = ""
    resultados['SOLD'] = False
    resultados.set_index('Lote', inplace=True)
    resultados.drop(columns='GPT 90%', inplace=True)

    return resultados

# def carrega_parms(folder_path):
#     with open(os.path.join(folder_path,'parms.txt'), 'r') as f:
#         params = {}
#         for line in f:
#             key, value = line.strip().split(': ')
#             params[key] = value
#     return params

def carrega_parms(parms_file):
    file_parms = parms_file['parms.json']
    params = json.load(file_parms)      
    return params


def indica_parms_slider(params):

    ## Filtro Res
    option_res = st.selectbox(
        'Selecione criterio de escolha:',
        ('abaixo', 'acima'), index = [1 if str(params.get('option_res')) == 'acima' else 0][0])

    # Solicita ao usuário os parâmetros
    slider_bales_before = st.slider(f'Resistência {option_res} de:', 20, 40, int(params['slider_bales_before']))
    st.write(f'Resistência {option_res} de:', slider_bales_before)

    ## Filtro Mic
    # option_mic = st.selectbox(
    #     'Selecione criterio de escolha:',
    #     ('abaixo', 'acima'), key='mic', index = [1 if str(params.get('option_mic')) == 'acima' else 0][0]
    #     )

    slider_mic = st.slider(
        f'Mic entre:',
        2.00, 5.00, (float(params['slider_mic'][0]), float(params['slider_mic'][1])))
    st.write(f"Mic entre {float(slider_mic[0])} e {float(slider_mic[1])}")

    ## Filtro UHM
    option_uhm = st.selectbox(
        'Selecione criterio de escolha:',
        ('abaixo', 'acima'), key='uhm', index = [1 if str(params.get('option_uhm')) == 'acima' else 0][0])

    slider_uhm = st.slider(f'UHM {option_uhm} de:', 0.0, 3.0, float(params['slider_uhm']))
    st.write(f'UHM {option_uhm} de:', slider_uhm)
    return slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm

def processa_resultado(df,slider_bales_before, option_res,
                slider_mic, slider_uhm, option_uhm, resumo_file, rec_parm):
    resultado = stats_table(df,slider_bales_before, option_res,
                slider_mic, slider_uhm, option_uhm)
    if rec_parm == 1:
        
        rig = pd.read_excel(resumo_file["resumo.xlsx"])[['Lote','OFFERED','SOLD']].set_index('Lote')
        
        # Leitura e exibição do conteúdo do dataframe editável
        resultado1 = resultado.reset_index()
        resultado1['Lote'] = resultado1['Lote'].astype('int64')
        resultado2 = pd.merge(resultado1.drop(columns=['OFFERED','SOLD']),rig.reset_index(), how='left', on = 'Lote')
        resultado2.set_index('Lote',inplace=True)
    else:
        resultado2 = resultado
    
    # edited_df = st.data_editor((resultado2.reset_index(drop=False)))
    edited_df = st.data_editor(resultado2.reset_index(drop=False).set_index('Lote',drop=False))
    return edited_df, resultado2



# def salva_parms(folder_path, slider_bales_before, option_res,
#                 slider_mic, slider_uhm, option_uhm):
#     params = {}
#     # carrega os parametro no dic params
#     params['slider_bales_before'] = slider_bales_before
#     params['slider_mic'] = list(slider_mic)
#     params['slider_uhm'] = slider_uhm
#     params['option_res'] = option_res
#     params['option_uhm'] = option_uhm
    
#     with open(f'{folder_path}/parms.json', 'w') as json_file:
#         json.dump(params, json_file, indent=4)

def salva_parms(slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm):
    params = {
        'slider_bales_before': slider_bales_before,
        'slider_mic': list(slider_mic),
        'slider_uhm': slider_uhm,
        'option_res': option_res,
        'option_uhm': option_uhm
    }
    
    # Retorna os dados JSON como uma string
    return json.dumps(params, indent=4)

# Função auxiliar para converter DataFrame para Excel
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


def salva_resultado2(edited_df, df_resultado, slider_bales_before, option_res,
                slider_mic, slider_uhm, option_uhm):
    
    st.download_button(label='Download Excel',
                    data=to_excel(edited_df),
                    file_name='resumo.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key="resumo")

    # Chama a função modificada para obter os dados JSON
    dados_json = salva_parms(slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm)
    
    # Cria o botão de download
    st.download_button(
        label="Download parms.json",
        data=dados_json,
        file_name="parms.json",
        mime='application/json',
        key="parms"
    )



def solicita_parms_slider():
    ## Filtro Res
    option_res = st.selectbox(
        'Selecione criterio de escolha:',
        ('acima', 'abaixo'))

    # Solicita ao usuário os parâmetros
    slider_bales_before = st.slider(f'Resistência {option_res} de:', 20, 40, 28)
    st.write(f'Resistência {option_res} de:', slider_bales_before)

    
    ## Filtro Mic
    # option_mic = st.selectbox(
    #     'Selecione criterio de escolha:',
    #     ('abaixo', 'acima', 'entre'), key='mic')

    slider_mic = st.slider(
        f'Mic entre:', 
        2.00, 5.00, (3.70, 4.90))
    # st.write(f'Mic entre {slider_mic[0]} e {slider_mic[1]}')
    st.write(f"Mic entre {float(slider_mic[0])} e {float(slider_mic[1])}")
    ## Filtro UHM
    option_uhm = st.selectbox(
        'Selecione criterio de escolha:',
        ('acima', 'abaixo'), key='uhm')

    slider_uhm = st.slider(f'UHM {option_uhm} de:', 0.00, 3.00, 1.11)
    st.write(f'UHM {option_uhm} de:', slider_uhm)
    return slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm


def gera_df(files):
    
    # Extração dos arquivos xlsx
    df = run_extract_table(files)
    return df


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


def func_sliders(rec_parm,params):
    if rec_parm == 1:
        ## Rodar indica_parms_slider...
        st.header("Filtrando dos Lotes:")
        slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm = indica_parms_slider(params)
                    
    else:
        # Solicita ao usuário os parâmetros
        st.title("Informe os parâmetros")
        slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm = solicita_parms_slider()
    
    return slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm

def check_params(parms_file):
    # Verifica a existência do arquivo parms.txt e se está vazio
    if parms_file != {}:
        st.title("Parâmetros previamente escolhidos")
        # lê e exibe os parâmetros
        params = carrega_parms(parms_file)
        st.write(params)
        rec_parm = 1
        
    ## Se não existe params.txt o usuario indica os params.    
    else:
        st.title("Arquivo 'parms.json' não encontrado.")
        params = {}
        rec_parm = 0

    return rec_parm, params



def carrega_logo():
    # Inserindo o logo   
    imagem_local = "https://www.vequis.com.br/assets/img/logo_vequis_white.svg"
    st.image(imagem_local, caption='', width=400,use_column_width=False)

# Função principal
def main():
    st.set_page_config(layout="wide") 
    carrega_logo()
    st.title("HVI Analysis System")
    
    st.header("Selecione os Lotes:")
    xlsx_files, resumo_file, parms_file = selecionar_lotes()

    # if os.path.exists(folder_path) and os.path.isdir(folder_path):
    if xlsx_files != {}:        
        st.success(f"Carregando os Lotes!")
        ## Checa se tem params anteriores
        rec_parm, params = check_params(parms_file)

        ## Roda sliders        
        slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm = func_sliders(rec_parm,params)
        
        ## Gera df
        df = gera_df(xlsx_files)
    
        ## Processando os arquivos e gerando a tabela resultado
        st.header("Resumo dos Lotes:")
        edited_df, df_resultado = processa_resultado(df,slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm, resumo_file, rec_parm)
        
        st.header("Salvar Resumo dos Lotes:")      
        # salva_resultado2(df_resultado, params, slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm, folder_path)
        salva_resultado2(edited_df, df_resultado, slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm)
            
    else:
        st.error(f"Por favor selecione um Lote válido.")


# Executa o aplicativo
if __name__ == "__main__":
    main()



