import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import io
import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import tempfile
from streamlit_pdf_viewer import pdf_viewer as view
from datetime import datetime
from dotenv import load_dotenv
import bcrypt

load_dotenv()

def checar_senha(senha_hashed, senha_usuario):
    return bcrypt.checkpw(senha_usuario.encode('utf-8'), senha_hashed)


def tela_login():
    st.title("Login")
    usuario = st.text_input("Usuário")
    senha = st.text_input('Senha', type='password')

    usuario_armazenado = os.getenv('Usuario')
    senha_armazenada = os.getenv('PASSWORD_HASHED')

    if senha_armazenada:
        senha_armazenada = senha_armazenada.encode('utf-8')

    if st.button('Entrar'):
        if usuario == usuario_armazenado and checar_senha(senha_armazenada, senha):
            st.session_state.logged_in = True
            st.success('Logado com sucesso')
        else:
            st.error('Usuário ou senha incorretos')


warnings.filterwarnings("ignore")

def tela_logado():
    def extract_table_from_file(path_to_file):
        # st.write(path_to_file.name)
        if path_to_file.name != 'resumo.xlsx':
            # st.write(path_to_file.name)
            df = pd.read_excel(path_to_file)
            
            indice_inicio = df.head(7).T.isna().sum().index[df.head(7).T.isna().sum()<4][0]
            cols = df.iloc[indice_inicio].to_list()
            try:
                if 'UHM' in cols:
                    cols = cols
                elif 'UHM' in df.columns.tolist():
                    cols = df.columns.tolist()
                else:
                    raise ValueError("A coluna 'UHM' não foi encontrada em cols nem em df.columns.tolist()")
            except Exception as e:
                print(e)

            cols = ['NaN' if pd.isna(valor) else valor for valor in cols]
            # print(cols)
            df = df.set_axis(cols, axis = 1)
            drop_indices = df.head(7).T.isna().sum().index[df.head(7).T.isna().sum()>4].tolist()
            # Removendo as linhas com base nos índices
            df = df.drop(drop_indices).iloc[1:] #, inplace=True)
            df = df.reset_index(drop=True)
            # indice_final = df[df["Mic"].isna()].index[0]
            # indice_final = df[df["UHM"].isna()].index[0]
            try:
                indice_final = df[df["UHM"].isna()].index[0]
            except IndexError:
                # Define indice_final como a quantidade de linhas de df caso o erro ocorra
                indice_final = len(df)


            df =  df.drop(df.index[indice_final:])
            lote = path_to_file.name.replace('.xlsx', '')
            df['Lote'] = lote
            # lote = df['Lote'].str.replace('/', '', regex=False)
            # df['Lote'] = lote   
            # df['COR'] = '31-4'
        

            # Verifique se a coluna 'LEAF' existe no DataFrame
            if 'LEAF' not in df.columns:
                # Se não existir, crie uma coluna 'LEAF' com valores vazios
                df['LEAF'] = 0

            # Verifique se a coluna 'COR' existe no DataFrame
            if 'COR' not in df.columns:
                # Se não existir, crie uma coluna 'COR' com valores vazios
                df['COR'] = '0'

            # Verifique se a coluna 'P. Líquido' existe no DataFrame
            if 'P. Líquido' not in df.columns:
                # Se não existir, crie uma coluna 'P. Líquido' com valores vazios
                df['P. Líquido'] = 0

            # Verifique se a coluna 'Mic' existe no DataFrame
            if 'Mic' not in df.columns:
                # Se não existir, crie uma coluna 'Mic' com valores vazios
                df['Mic'] = 0

            # Verifique se a coluna 'UHM' existe no DataFrame
            if 'UHM' not in df.columns:
                # Se não existir, crie uma coluna 'UHM' com valores vazios
                df['UHM'] = 0

            # Verifique se a coluna 'Res' existe no DataFrame
            if 'Res' not in df.columns:
                # Se não existir, crie uma coluna 'Res' com valores vazios
                df['Res'] = 0            

            sel_cols = ['Lote','Fardo','P. Líquido', 'Mic', 'UHM', 'Res', 'COR', 'LEAF']
            df = df[sel_cols]
            
            
            # df['COR'] = df['COR'].str.replace('"','')

            # df['Res'] = df['Res'].astype(float)
            # df['Res'].dropna(inplace=True)

            # if np.count_nonzero(df.Res) == 0:
            #     st.write(path_to_file.name)
            

        else:
            df = pd.read_excel(path_to_file)

        return df

    def extract_table_from_file_secon(path_to_file):
        # st.write(path_to_file.name)
        if path_to_file.name != 'resumo.xlsx':
            # st.write(path_to_file.name)
            df = pd.read_excel(path_to_file)
            
            indice_inicio = df.head(7).T.isna().sum().index[df.head(7).T.isna().sum()<4][0]
            cols = df.iloc[indice_inicio].to_list()
            try:
                if 'UHM' in cols:
                    cols = cols
                elif 'UHM' in df.columns.tolist():
                    cols = df.columns.tolist()
                else:
                    raise ValueError("A coluna 'UHM' não foi encontrada em cols nem em df.columns.tolist()")
            except Exception as e:
                print(e)

            cols = ['NaN' if pd.isna(valor) else valor for valor in cols]
            # print(cols)
            df = df.set_axis(cols, axis = 1)
            drop_indices = df.head(7).T.isna().sum().index[df.head(7).T.isna().sum()>4].tolist()
            # Removendo as linhas com base nos índices
            df = df.drop(drop_indices).iloc[1:] #, inplace=True)
            df = df.reset_index(drop=True)
            # indice_final = df[df["Mic"].isna()].index[0]
            # indice_final = df[df["UHM"].isna()].index[0]
            try:
                indice_final = df[df["UHM"].isna()].index[0]
            except IndexError:
                # Define indice_final como a quantidade de linhas de df caso o erro ocorra
                indice_final = len(df)


            df =  df.drop(df.index[indice_final:])
            lote = path_to_file.name.replace('.xlsx', '')
            df['Lote'] = lote
            # lote = df['Lote'].str.replace('/', '', regex=False)
            # df['Lote'] = lote   
            # df['COR'] = '31-4'
        

            # Verifique se a coluna 'LEAF' existe no DataFrame
            if 'LEAF' not in df.columns:
                # Se não existir, crie uma coluna 'LEAF' com valores vazios
                df['LEAF'] = 0

            # Verifique se a coluna 'COR' existe no DataFrame
            if 'COR' not in df.columns:
                # Se não existir, crie uma coluna 'COR' com valores vazios
                df['COR'] = '0'

            # Verifique se a coluna 'P. Líquido' existe no DataFrame
            if 'P. Líquido' not in df.columns:
                # Se não existir, crie uma coluna 'P. Líquido' com valores vazios
                df['P. Líquido'] = 0

            # Verifique se a coluna 'Mic' existe no DataFrame
            if 'Mic' not in df.columns:
                # Se não existir, crie uma coluna 'Mic' com valores vazios
                df['Mic'] = 0

            # Verifique se a coluna 'UHM' existe no DataFrame
            if 'UHM' not in df.columns:
                # Se não existir, crie uma coluna 'UHM' com valores vazios
                df['UHM'] = 0

            # Verifique se a coluna 'Res' existe no DataFrame
            if 'Res' not in df.columns:
                # Se não existir, crie uma coluna 'Res' com valores vazios
                df['Res'] = 0  
            if 'Máquina' not in df.columns:
                df['Máquina'] = 0
            df['Aplicação'] = "Negado"
            
            if df['Máquina'].isnull().all():
                df['Máquina'] = df['NaN']   
        
            sel_cols = ['Lote','Fardo','P. Líquido', 'Máquina', 'Mic', 'UHM', 'Res', 'COR', 'LEAF', 'Aplicação']
            df = df[sel_cols]
            #st.dataframe(df)  
            # df['COR'] = df['COR'].str.replace('"','')

            # df['Res'] = df['Res'].astype(float)
            # df['Res'].dropna(inplace=True)

            # if np.count_nonzero(df.Res) == 0:
            #     st.write(path_to_file.name)
            

        else:
            df = pd.read_excel(path_to_file)

        return df

    def alerta(lote,lista):
        # Inicializamos uma string vazia para armazenar os números
        numeros_frase = ""
        # Iteramos pela lista e adicionamos os números à string
        for i, num in enumerate(lista):
            if i < len(lista) - 2:
                numeros_frase += str(num) + ", "
            elif i == len(lista) - 2:
                numeros_frase += str(num) + " e "
            else:
                numeros_frase += str(num)

        # Criamos a frase completa
        observacao = f"""O Lote {lote} apresenta os seguintes fardos duplicados {numeros_frase} .
        O usuário deve alterar manuamente este Lote."""
        if lista != []:
            st.warning("🚨 Atenção!! 🚨")
            st.warning(observacao)



    def run_extract_table(files):
        
        # Itere sobre cada arquivo
        lotes_duplicados = []
        df = pd.DataFrame()
        for file in files:    
            
            table = extract_table_from_file(file)
            
            if file.name != 'resumo.xlsx':
                if not table['Fardo'].duplicated().any():
                    # print("Lote sem Fardos duplicados")
                    df =  pd.concat([df,table])
                    # st.header("Debugg11")
                    # st.write(df.head())
                else:
                    lotes_duplicados.append(table['Lote'].unique()[0]) 
                    fardos_duplicados = table[table['Fardo'].duplicated()]['Fardo'].unique().tolist()
                    lotes_duplicado = table['Lote'].unique()[0]
                    alerta(lotes_duplicado,fardos_duplicados)
                    # df =  pd.concat([df,table])   
                    # st.header("Debugg22")
                    # st.write(lotes_duplicados)
                    # st.write(table['Fardo'].duplicated())
                
            else:
                df = table
        # st.header("Debugg")
        # st.write(df.head())

        return df, lotes_duplicados

    def stats_table(df,slider_bales_before=28, option_res= 'acima',
                    slider_mic = (3.58,4.5),
                    slider_uhm=1.10, option_uhm= 'acima'):
        # tratamento da Cor
        # st.write(df.head())
        df['COR'] = df['COR'].str.replace('"', '')
        df['COR'] = df['COR'].str.split('-').str[0]
        # df['COR'] = df['COR'].fillna(0)
        df.dropna(inplace=True)
        # Agrupa por 'lote' e calcula a estatistica 
        df['UHM'] = df['UHM'].astype(float)
        df['Res'] = df['Res'].astype(float)
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
        edited_df = st.data_editor(resultado2.reset_index(drop=False).set_index('Lote',drop=False), disabled=['Lote', 'Net weight', 'Mic(avg)', 'Mic(min)', 'Mic(max)'])
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
        df, lotes_duplicados = run_extract_table(files)
        return df, lotes_duplicados

        # loop que percorre a lista de arquivos passados e verifica e mostra o arquivo que possui o nome da opção selecionada

    def selecionar_lotes():
        uploaded_files = st.file_uploader("Faça upload dos arquivos: ", accept_multiple_files=True, type=["xlsx","json"])

        if uploaded_files != {}:
            # Dicionários para separar os arquivos
            xlsx_files = []
            resumo_file = []
            parms_file = []

            nomes_arquivos = []

            for idx, uploaded_file in enumerate(uploaded_files):
                # Verifica se é o arquivo parms.json
                if uploaded_file.name == 'parms.json':
                    #parms_file[uploaded_file.name] = uploaded_file
                    parms_file.append(uploaded_file)
                # Verifica se é o arquivo resumo.xlsx
                elif uploaded_file.name == 'resumo.xlsx':
                    # resumo_file[uploaded_file.name] = uploaded_file
                    resumo_file.append(uploaded_file)
                # Todos os outros arquivos .xlsx
                elif uploaded_file.name.endswith('.xlsx') and uploaded_file.name != 'resumo.xlsx':
                    # xlsx_files[int(idx)] = uploaded_file
                    xlsx_files.append(uploaded_file)
                nomes_arquivos.append(uploaded_file.name)
        return xlsx_files, resumo_file, parms_file


    def atualizar_planilha(df, caixaSelecao):
        new_df = df
        lote_exibido = df[df['Lote'] == caixaSelecao]
        lote_editado = st.data_editor(lote_exibido, hide_index=True, disabled=['Lote', 'Fardo', 'P. Líquido', 'Mic', 'UHM', 'Res', 'COR', 'LEAF'])
        df[df['Lote'] == caixaSelecao] = lote_editado

        return new_df

    def gerar_df_geral(arquivos_lotes):
        df = pd.DataFrame()
        for arquivo in arquivos_lotes:
            tabela = extract_table_from_file_secon(arquivo)
            df = pd.concat([df, tabela], ignore_index=True)
        if 'df_geral' not in st.session_state:
            st.session_state.df_geral = df
        
        return st.session_state.df_geral

    def selecioneOLote(lista_lotes, df): 
        opcoes = ['Selecione um lote'] + lista_lotes.Lote.to_list()
        caixaSelecao = st.selectbox('Visualizar lote', opcoes)
        
        if caixaSelecao != 'Selecione um lote':
            st.write(f'Expandir lote {caixaSelecao}')
            lote_exibido = st.session_state.df_geral[st.session_state.df_geral['Lote'] == caixaSelecao]
            lote_editado = st.data_editor(lote_exibido, hide_index=True, disabled=['Lote', 'Fardo', 'P. Líquido', 'Mic', 'UHM', 'Res', 'COR', 'LEAF', 'Máquina'])
            st.session_state.df_geral.loc[st.session_state.df_geral['Lote'] == caixaSelecao] = lote_editado
        else:
            st.write("Nenhum lote selecionado.")


    def selecionar_pelo_contrato(df_geral):
        opcoes = df_geral['Aplicação'].unique()
        contrato_escolhido = st.selectbox('Selecione uma aplicação', opcoes)
        df_contrato = df_geral[df_geral['Aplicação'] == contrato_escolhido]
        st.data_editor(df_contrato, hide_index=True, disabled=['Lote', 'Fardo', 'P.Líquido', 'Mic', 'UHM', 'Res', 'COR', 'LEAF', 'Máquina'])

    def mmToP(mm):
        return mm * 2.83465

    def gerar_formulario():
        st.markdown("""<style>
                    .stTextInput {
                        width: 400px
                    }
                    </style>""",
                    unsafe_allow_html=True
                    )
        
        col1, col2 = st.columns(2)
        with col1:
            instrucao = st.text_input('Instrução')
            filial = st.text_input('Filial')
            codigo = st.text_input('Código')
            veiculo = st.text_input('Veículo')
            nota_fiscal = st.text_input('Nota Fiscal')
            pedvenda = st.text_input('Ped. Venda')
        with col2:
            empresa = st.text_input('Empresa')
            remente = st.text_input('Rementente')
            destinatario = st.text_input('Detinatário')
            enedereco = st.text_input('Endereço')
            bloco = st.text_input('Bloco')
            data_saida = st.date_input('Data Saída', format=('DD/MM/YYYY'))

        return instrucao,filial,codigo, veiculo, nota_fiscal, empresa, remente, destinatario, enedereco, pedvenda, data_saida, bloco
    
    def centralizar_texto_tabela(c, texto, largura_celula, x, y):
        largura_texto = c.stringWidth(texto, "Helvetica", 6)
        x += (largura_celula - largura_texto) / 2 
        c.drawString(x, y, texto)
        #c.drawString(x, y, texto)

    def alinhar_texto(txt, c, instrucao, tamanho, altura, posicao):
        if posicao == 'direita':
            text_to_right = f'{txt}: {instrucao}'
            margem_esquerda = mmToP(203) - (c.stringWidth(text_to_right,'Helvetica', tamanho))
            c.drawString(margem_esquerda, altura, f'{text_to_right}')
        else:
            text_to_center = f'{txt}: {instrucao}'
            margem_esquerda = (mmToP(210) - c.stringWidth(text_to_center,'Helvetica', tamanho)) / 2
            c.drawString(margem_esquerda, altura, f'{text_to_center}')

    def juntar_dado(c, txt1, txt2, template1, template2, altura, tamanho):
        primeiro_texto = f'{txt1}: {template1}'
        margem = (c.stringWidth(primeiro_texto, 'Helvetica', tamanho)) + mmToP(20)
        c.drawString(mmToP(7), altura, primeiro_texto)
        c.drawString(margem, altura, f'{txt2}: {template2}')

    def alinha_celula_somatoria(c, texto, largura_celula, numero, y, tamanho, posicao):
        if posicao == 'centro':
            numero -= 1
            largura_texto = c.stringWidth(texto ,'Helvetica', tamanho)
            margem_esquerda = mmToP(7) + numero * largura_celula + (largura_celula - largura_texto) / 2
            c.drawString(margem_esquerda, y, texto)
        if posicao == 'direita':
            #numero -= 1
            altura_texto = y - mmToP(5)
            largura_texto = c.stringWidth(texto ,'Helvetica', tamanho)
            margem_esquerda = mmToP(7) + largura_celula * numero - largura_texto
            c.drawString(margem_esquerda, altura_texto, texto)


            


    @st.cache_data
    def gerar_pdf(contrato_escolhido, tabela_contrato, instrucao,filial,codigo, veiculo, nota_fiscal, empresa, remente, destinatario, enedereco, pedvenda, data_saida, bloco):
        data_formatada = data_saida.strftime('%d/%m/%Y')
        tabela_pdf = tabela_contrato[['Fardo', 'COR', 'P. Líquido', 'Máquina']]
        tabela_pdf.columns = ['Fardo', 'Tipo', 'P. Líquido', 'Prensa']
        st.dataframe(tabela_pdf, hide_index=True)
        hora_atual = datetime.now().strftime('%H:%M:%S')
        data_atual = datetime.now().strftime("%d/%m/%Y")
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        # imagem
        c.drawImage('logoCR.png', mmToP(7), mmToP(283), width=30, height=15)

        # strings tamanho 6
        c.setFont('Helvetica', 6)
        c.drawString(mmToP(7), mmToP(280), 'SIGA / AGRAR750 / v. 12', )
        c.drawString(mmToP(7), mmToP(276.5), f'Hora . . .: {hora_atual}')
        c.drawString(mmToP(7), mmToP(273), f'Empresa: {empresa}')
        c.drawString(mmToP(7), mmToP(269.5), f'Filial: {filial}')
        c.drawString(mmToP(7), mmToP(264), f'Remetente: {remente}')
        c.drawString(mmToP(7), mmToP(260.5), f'Código: {codigo}')
        c.drawString(mmToP(7), mmToP(250), f'Contrato: {contrato_escolhido}')
        c.drawString(mmToP(7), mmToP(246.5), f'Ped. Venda: {pedvenda}')
        c.drawString(mmToP(7), mmToP(237), f'Bloco: {bloco}')

        #textos na mesma linha
        juntar_dado(c, 'Destinatário', 'Endereço', destinatario, enedereco, mmToP(257), 6)
        juntar_dado(c, 'Veículo', 'Data Saída', veiculo, data_formatada, mmToP(253.5), 6)
        #juntar_dado(dado1, c, 'Endereço', enedereco, mmToP(257), 6)

        #textos alinhados (centro ou direita)
        alinhar_texto('Folha . . .', c, 1, 6, mmToP(280), 'direita')
        alinhar_texto('Dt. Emissão', c, data_atual, 6, mmToP(276.5), 'direita')
        alinhar_texto('Romaneio de Saída de Fardos', c, '', 6, mmToP(280),'centro')

        c.setFont('Helvetica', 8)
        alinhar_texto('Hora', c, hora_atual, 8, mmToP(12.5), 'direita')

        #texto tamanho 15 alinhado
        c.setFont('Helvetica-Bold', 15)
        alinhar_texto('Instrução', c, instrucao, 15, mmToP(273), 'centro')

        c.setFont('Helvetica-Bold', 6)
        c.drawString(mmToP(7), mmToP(243), f'OBS:Este romaneio é parte integrante da nota fiscal {nota_fiscal}')

        # linhas
        c.setLineWidth(0.5)
        c.line(mmToP(7), mmToP(289), mmToP(203), mmToP(289))
        c.line(mmToP(7), mmToP(268), mmToP(203), mmToP(268))
        c.line(mmToP(7), mmToP(241), mmToP(203), mmToP(241))
        c.line(mmToP(7), mmToP(16), mmToP(203), mmToP(16))
        c.line(mmToP(7), mmToP(11), mmToP(203), mmToP(11))

        #tabela
        num_colunas = len(tabela_pdf.columns)
        num_linhas = len(tabela_pdf)
        x_inicial = mmToP(7)
        y_inicial = mmToP(230)
        largura_celula = mmToP(12.24)
        altura_celula = mmToP(2.7)
        y = y_inicial
        conjuntos_por_linha = 4
        x = x_inicial

        for coluna in tabela_pdf.columns:
            for coluna in tabela_pdf.columns: 
                c.rect(x, y, largura_celula, altura_celula, fill=False) 
                #c.drawString(x + 2, y + 2, str(coluna))
                c.setFont('Helvetica-Bold', 6)
                centralizar_texto_tabela(c, str(coluna), largura_celula, x, y + 1.8)

                x += largura_celula 

        y -= altura_celula

        c.setFont('Helvetica', 6)

        for linha in range(0, num_linhas, conjuntos_por_linha):  
            x = x_inicial

            for i in range(conjuntos_por_linha):
                if linha + i < num_linhas:
                    dados_linha = tabela_pdf.iloc[linha + i]
                    for coluna in range(num_colunas):
                        dado = dados_linha[coluna]
                        c.rect(x, y, largura_celula, altura_celula, fill=False)
                        centralizar_texto_tabela(c, str(dado), largura_celula, x, y + 1.8)
                        #c.drawString(x + 2, y + 2, str(dado)) 
                        x += largura_celula

                else:
                    for coluna in range(num_colunas):
                        c.rect(x, y, largura_celula, altura_celula, fill=False)  
                        x += largura_celula

            y -= altura_celula
            x = x_inicial

        # calculos e blocos
        altura_inical = y - mmToP(5)
        tamanho_celula = mmToP(18)
        peso_total_bloco = tabela_pdf['P. Líquido'].sum()
        qtd_fardos = len(tabela_pdf)
        tipo = tabela_pdf['Tipo'][0]
        alinha_celula_somatoria(c, 'Bloco', tamanho_celula, 1, altura_inical, 9, 'centro')
        alinha_celula_somatoria(c, 'Qtd de blocos', tamanho_celula, 2, altura_inical, 9, 'centro')
        alinha_celula_somatoria(c, 'Tipo', tamanho_celula, 3, altura_inical, 9, 'centro')
        alinha_celula_somatoria(c, 'Peso Total', tamanho_celula, 4, altura_inical, 9, 'centro')
        alinha_celula_somatoria(c, f'{bloco}', tamanho_celula, 1, altura_inical, 9, 'direita')
        alinha_celula_somatoria(c, f'{qtd_fardos}', tamanho_celula, 2, altura_inical, 9, 'direita')
        alinha_celula_somatoria(c, f'{tipo}', tamanho_celula, 3, altura_inical, 9, 'direita')
        alinha_celula_somatoria(c, f'{peso_total_bloco}', tamanho_celula, 4, altura_inical, 9, 'direita')
        c.drawString(mmToP(7), altura_inical - mmToP(11), f'Peso total: {peso_total_bloco}')
        c.drawString(mmToP(7), altura_inical - mmToP(14), f'Fardos Total: {qtd_fardos}')
        

        c.line(mmToP(7), altura_inical - mmToP(1), tamanho_celula * 4 + mmToP(7), altura_inical - mmToP(1))
        c.line(mmToP(7), altura_inical - mmToP(7), tamanho_celula * 4 + mmToP(7), altura_inical - mmToP(7))


        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    def selecao_contratos(df):
        contratos = df['Aplicação'].unique()  
        caixaSelecao = st.selectbox('Visualizar lote', contratos)

        if caixaSelecao != 'Negado':
            contrato_escolhido = df[df['Aplicação'] == caixaSelecao]
            instrucao,filial,codigo, veiculo, nota_fiscal, empresa, remente, destinatario, enedereco, pedvenda, data_saida, bloco = gerar_formulario()
            pdf_buffer = gerar_pdf(caixaSelecao, contrato_escolhido, instrucao,filial,codigo, veiculo, nota_fiscal, empresa, remente, destinatario, enedereco, pedvenda, data_saida, bloco)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_buffer.read())
                tmp_pdf_path = tmp_pdf.name
            view(tmp_pdf_path)
            with open(tmp_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button('Salvar contrato', data=pdf_bytes, file_name=f'{caixaSelecao}.pdf')

        else: 
            st.write('Selecione um contrato válido')
            
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
        if parms_file != []:
            st.title("Parâmetros previamente escolhidos")
            # lê e exibe os parâmetros
            params = carrega_parms(parms_file)
            st.write(params)
            rec_parm = 1
            
        ## Se não existe params.txt o usuario indica os params.    
        else:
            st.title("Arquivo 'parms.json' não encontrado.")
            params = []
            rec_parm = 0

        return rec_parm, params



    def carrega_logo():
        # Inserindo o logo   
        imagem_local = "https://www.vequis.com.br/assets/img/logo_vequis_white.svg"
        st.image(imagem_local, caption='', width=400,use_column_width=False)

    # minha funcao


    # Função principal
    def main():
        st.set_page_config(layout="wide") 
        carrega_logo()
        st.title("HVI Analysis System")
        
        st.header("Selecione os Lotes:")
        xlsx_files, resumo_file, parms_file = selecionar_lotes()
        # if os.path.exists(folder_path) and os.path.isdir(folder_path):
        if xlsx_files != []:        
            st.success(f"Carregando os Lotes!")
            ## Checa se tem params anteriores
            rec_parm, params = check_params(parms_file)

            ## Roda sliders        
            slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm = func_sliders(rec_parm,params)
            
            ## Gera df
            df, lotes_duplicados = gera_df(xlsx_files)
            if lotes_duplicados != []:
                st.warning(f"Atenção Fardos com mais de uma medida no lote {lotes_duplicados}")
                # alerta(lotes_duplicados)

            if df.shape[0] > 0:
            
                ## Processando os arquivos e gerando a tabela resultado
                st.header("Resumo dos Lotes:")
                edited_df, df_resultado = processa_resultado(df,slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm, resumo_file, 
                rec_parm)

                st.header("Salvar Resumo dos Lotes:")      
                # salva_resultado2(df_resultado, params, slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm, folder_path)
                salva_resultado2(edited_df, df_resultado, slider_bales_before, option_res, slider_mic, slider_uhm, option_uhm)
                
                # Gerar df geral
                df_geral = gerar_df_geral(xlsx_files)

                st.header("Emblocagem:")  
                opcao = st.radio('Selecione a tabela por:', ['Lote', 'Contrato'], index=None)
                if opcao == 'Lote':
                    selecioneOLote(edited_df, df_geral)
                elif opcao == 'Contrato': 
                    selecionar_pelo_contrato(df_geral)
                else:
                    st.write('Nenhuma opção selecionada')


                st.header("Contratos e Romaneios:") 
                st.subheader('Selecione o contrato')
                selecao_contratos(df_geral)
        else:
            st.error(f"Por favor selecione um Lote válido.")

        

    # Executa o aplicativo
    if __name__ == "__main__":
        main()


if 'logged_in' not in st.session_state:
    st.session_state.logged_in=False

if st.session_state.logged_in:
    tela_logado()
else: tela_login()

