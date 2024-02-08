import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import io
from openpyxl import Workbook

warnings.filterwarnings("ignore")


def extract_table_from_file(path_to_file, folder_path):
    df = pd.read_excel(path_to_file)
    cols = df.iloc[2].to_list()
    cols = ['NaN' if pd.isna(valor) else valor for valor in cols]
    # print(cols)
    df = df.set_axis(cols, axis = 1)
    df = df.iloc[5:]
    indice_final = df[df["Mic"].isna()].index[0]
    df =  df.drop(df.index[indice_final-5:])
    # lote = path_to_file.replace(os.getcwd()+'/', '').replace('.xlsx', '')
    lote = path_to_file.replace(path_to_file+'/', '').replace('.xlsx', '')
    df['Lote'] = lote
    lote = df['Lote'].str.replace(folder_path+'/', '', regex=False)
    df['Lote'] = lote   
    df['COR'] = '31-4'

    # Verifique se a coluna 'LEAF' existe no DataFrame
    if 'LEAF' not in df.columns:
        # Se não existir, crie uma coluna 'LEAF' com valores vazios
        df['LEAF'] = np.nan

    sel_cols = ['Lote','Fardo','P. Líquido', 'Mic', 'UHM', 'Res', 'COR', 'LEAF']
    df = df[sel_cols]

    return df



# Obtenha o caminho do diretório de trabalho atual
def catch_path_files(path_data):
    # Primeiro, vá para o diretório especificado
    # Liste todos os arquivos e diretórios no caminho especificado
    path_to_folder = path_data
    files_and_folders = os.listdir(path_to_folder)
    # Filtra apenas os arquivos com extensão .xlsx e exclui "resumo.xlsx"
    files = [os.path.join(path_to_folder, f) for f in files_and_folders if os.path.isfile(os.path.join(path_to_folder, f)) and f.endswith('.xlsx') and f != 'resumo.xlsx']
    
    return files

def run_extract_table(files, folder_path):
    # Itere sobre cada arquivo
    df = pd.DataFrame()
    for file in files:    
        # Define o caminho do arquivo 
        path_to_file = file
        # print(path_to_file)
        table = extract_table_from_file(path_to_file, folder_path)
        df =  pd.concat([df,table])

    return df

def stats_table(df,slider_bales_before=28, option_res= 'acima',
                # slider_mic_min=3.86, slider_mic_max=4.50, 
                slider_mic = (3.58,4.5),
                slider_uhm=1.10, option_uhm= 'acima'):
                # slider_mic=df.Mic.mean().round(2), option_mic= 'acima',
                # slider_uhm=df.UHM.mean().round(2), option_uhm= 'acima',):
    # Agrupa por 'lote' e calcula a estatistica 
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
        Res = df.groupby('Lote').agg(Bales_below_28=pd.NamedAgg(column='Res', aggfunc=lambda x: np.count_nonzero(x>slider_bales_before)/np.count_nonzero(x))).reset_index()
    elif option_res == 'abaixo':
        Res = df.groupby('Lote').agg(Bales_below_28=pd.NamedAgg(column='Res', aggfunc=lambda x: np.count_nonzero(x<slider_bales_before)/np.count_nonzero(x))).reset_index()

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
        UHM = df.groupby('Lote').agg(UHM_option=pd.NamedAgg(column='UHM', aggfunc=lambda x: np.count_nonzero(x>slider_uhm)/np.count_nonzero(x))).reset_index()
    elif option_uhm == 'abaixo':
        UHM = df.groupby('Lote').agg(UHM_option=pd.NamedAgg(column='UHM', aggfunc=lambda x: np.count_nonzero(x<slider_uhm)/np.count_nonzero(x))).reset_index()

    resultados = pd.merge(resultados,UHM,how='left',on='Lote')

    # Formatar a coluna como porcentagem   
    resultados['Bales_below_28'] = resultados['Bales_below_28'].map("{:.0%}".format)
    resultados['Mic_option'] = resultados['Mic_option'].map("{:.1%}".format)
    resultados['UHM_option'] = resultados['UHM_option'].map("{:.1%}".format)
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

    # # Renomeia as colunas
    resultados.columns = ['Lote', 'P. Líquido', 'Mic (avg)', 'Mic (min)', 'Mic (max)',
                            'UHM Avg', 'UHM min', 'UHM max',
                            'GPT avg', 'GPT min', 'GPT Max', 'GPT 90%', 
                            f'Res {option_res} de {slider_bales_before}',
                            'LEAF Avg',
                            f'Mic entre {slider_mic[0]} e {slider_mic[1]}',
                            f'UHM {option_uhm} de {slider_uhm}',]

    def class_uhm(valor):
        if 0 <= valor <= 0.79:
            return 24
        elif 0.80 <= valor <= 0.85:
            return 26
        elif 0.86 <= valor <= 0.89:
            return 28
        elif 0.90 <= valor <= 0.92:
            return 29
        elif 0.93 <= valor <= 0.95:
            return 30
        elif 0.96 <= valor <= 0.98:
            return 31
        elif 0.99 <= valor <= 1.01:
            return 32
        elif 1.02 <= valor <= 1.04:
            return 33
        elif 1.05 <= valor <= 1.07:
            return 34
        elif 1.08 <= valor <= 1.10:
            return 35
        elif 1.11 <= valor <= 1.13:
            return 36
        elif 1.14 <= valor <= 1.17:
            return 37
        elif 1.18 <= valor <= 1.20:
            return 38
        elif 1.21 <= valor <= 1.23:
            return 39
        elif 1.24 <= valor <= 1.26:
            return 40
        elif 1.27 <= valor <= 1.29:
            return 41
        elif 1.30 <= valor <= 1.32:
            return 42
        elif 1.33 <= valor <= 1.35:
            return 43
        elif valor >= 1.36:
            return 44
        else:
            return np.nan  # Classificação padrão para valores fora das faixas

    # Aplicar a função à coluna 'valor' e criar uma nova coluna 'classificacao'
    df['UHM_class'] = df['UHM'].apply(class_uhm)
    UHM_class = pd.crosstab(df.Lote,df.UHM_class).add_prefix("UHM_").reset_index()

    resultados = pd.merge(resultados,UHM_class,how='left',on='Lote')


    resultados['OFFERED'] = np.nan
    resultados['SOLD'] = False
    resultados.set_index('Lote', inplace=True)
    resultados.drop(columns='GPT 90%', inplace=True)

    return resultados

def carrega_parms(folder_path):
    with open(os.path.join(folder_path,'parms.txt'), 'r') as f:
        params = {}
        for line in f:
            key, value = line.strip().split(': ')
            params[key] = value
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
        2.00, 5.00, (float(eval(params['slider_mic'])[0]), float(eval(params['slider_mic'])[1])))
    st.write(f"Mic entre {float(slider_mic[0])} e {float(slider_mic[1])}")

    ## Filtro UHM
    option_uhm = st.selectbox(
        'Selecione criterio de escolha:',
        ('abaixo', 'acima'), key='uhm', index = [1 if str(params.get('option_uhm')) == 'acima' else 0][0])

    slider_uhm = st.slider(f'UHM {option_uhm} de:', 0.0, 3.0, float(params['slider_uhm']))
    st.write(f'UHM {option_uhm} de:', slider_uhm)
    return slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm

def processa_resultado(df,slider_bales_before, option_res,
                slider_mic, slider_uhm, option_uhm, folder_path, rec_parm):
    resultado = stats_table(df,slider_bales_before, option_res,
                slider_mic, slider_uhm, option_uhm)
    if rec_parm == 1:
        rig = pd.read_excel(f"{folder_path}/resumo.xlsx")[['Lote','OFFERED','SOLD']].set_index('Lote')
        
        # Leitura e exibição do conteúdo do dataframe editável
        resultado1 = resultado.reset_index()
        resultado1['Lote'] = resultado1['Lote'].astype('int64')
        resultado2 = pd.merge(resultado1.drop(columns=['OFFERED','SOLD']),rig.reset_index(), how='left', on = 'Lote')
        resultado2.set_index('Lote',inplace=True)
    else:
        resultado2 = resultado
    
    edited_df = st.data_editor((resultado2))
    # edited_df = st.dataframe((resultado2))
    return edited_df, resultado2

def salva_resultado2(df_resultado, params, slider_bales_before, option_res,
                slider_mic, slider_uhm, option_uhm, folder_path):
    
    df_resultado.to_excel(f"{folder_path}/resumo.xlsx")

    params['slider_bales_before'] = slider_bales_before
    params['slider_mic'] = slider_mic
    params['slider_uhm'] = slider_uhm
    params['option_res'] = option_res
    # params['option_mic'] = option_mic
    params['option_uhm'] = option_uhm

    # Salva os parâmetros em um arquivo
    with open(f"{folder_path}/parms.txt", 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")



def solicita_parms_slider():
    ## Filtro Res
    option_res = st.selectbox(
        'Selecione criterio de escolha:',
        ('abaixo', 'acima'))

    # Solicita ao usuário os parâmetros
    slider_bales_before = st.slider(f'Resistência {option_res} de:', 20, 40, 28)
    st.write(f'Resistência {option_res} de:', slider_bales_before)

    
    ## Filtro Mic
    # option_mic = st.selectbox(
    #     'Selecione criterio de escolha:',
    #     ('abaixo', 'acima', 'entre'), key='mic')

    slider_mic = st.slider(
        f'Mic entre:', 
        2.00, 5.00, (3.58, 4.5))
    # st.write(f'Mic entre {slider_mic[0]} e {slider_mic[1]}')
    st.write(f"Mic entre {float(slider_mic[0])} e {float(slider_mic[1])}")
    ## Filtro UHM
    option_uhm = st.selectbox(
        'Selecione criterio de escolha:',
        ('abaixo', 'acima'), key='uhm')

    slider_uhm = st.slider(f'UHM {option_uhm} de:', 0.00, 3.00, 1.11)
    st.write(f'UHM {option_uhm} de:', slider_uhm)
    return slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm

def gera_df(folder_path):
    files = catch_path_files(folder_path)
    if not files:
        st.warning("Nenhum arquivo xlsx encontrado na pasta fornecida.")
        st.stop()

    # Extração dos arquivos xlsx
    df = run_extract_table(files, folder_path)
    return df

def selecionar_lotes():
    # Caminho para a pasta principal
    # folder_path0 = "../../Dropbox/trading_app"
    folder_path0 = "data"

    # Lista de pastas disponíveis
    pastas_disponiveis = [nome for nome in os.listdir(folder_path0) if os.path.isdir(os.path.join(folder_path0, nome))]

    # Caixa de seleção para escolher a pasta
    pasta_escolhida = st.selectbox("Selecione uma Pasta", pastas_disponiveis)

    # Caminho completo para a pasta escolhida
    folder_path = os.path.join(folder_path0, pasta_escolhida)
    return folder_path

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

def check_params(folder_path):
    # Verifica a existência do arquivo parms.txt e se está vazio
    if os.path.exists(os.path.join(folder_path,'parms.txt')) and os.stat(os.path.join(folder_path,'parms.txt')).st_size != 0:
        st.title("Parâmetros previamente escolhidos")
        # lê e exibe os parâmetros
        params = carrega_parms(folder_path)
        st.write(params)
        rec_parm = 1
        
    ## Se não existe params.txt o usuario indica os params.    
    else:
        st.title("Arquivo 'parms.txt' não encontrado.")
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
    st.title("Trading App")
    carrega_logo()
    
    st.header("Selecione os Lotes:")
    folder_path = selecionar_lotes()

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        st.success(f"Caminho definido para: {folder_path}")
        ## Checa se tem params anteriores
        rec_parm, params = check_params(folder_path)

        ## Roda sliders        
        slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm = func_sliders(rec_parm,params)
        
        ## Gera df
        df = gera_df(folder_path)
    
        ## Processando os arquivos e gerando a tabela resultado
        st.header("Resumo dos Lotes:")
        edited_df, df_resultado = processa_resultado(df,slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm, folder_path, rec_parm)
        
        st.header("Salvar Resumo dos Lotes:")      
        # Botão para salvar o dataframe editado como resumo.xlsx e o ultimo params.txt
        if st.button("Salvar Resumo", key='first_save'):
            # salva_resultado2(df_resultado, params, slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm, folder_path)
            salva_resultado2(edited_df, params, slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm, folder_path)
                
    else:
        st.error(f"Por favor selecione um Lote válido.")


# Executa o aplicativo
if __name__ == "__main__":
    main()



