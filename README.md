# Dashboard de Trading de Algodão

Este projeto oferece um dashboard interativo desenvolvido com Streamlit e Python 3.10, destinado a empresas de trading de algodão. O dashboard facilita a análise rápida e precisa da qualidade e quantidade de fardos de algodão por Lote, baseando-se em informações chave como Resistência da Fibra, Mic e UHM. Usuários podem definir parâmetros para filtrar dados acima ou abaixo dos valores especificados para Resistência da Fibra, Mic e UHM, permitindo uma negociação mais ágil e informada. Além disso, é possível exportar uma tabela Resumo com as anotações de ofertas e vendas realizadas.

## Características Principais

- **Análise Interativa por Lote:** Visualize e interaja com os dados de cada lote de algodão, incluindo detalhes sobre a Resistência da Fibra, Mic e UHM.
- **Filtragem de Dados:** Defina parâmetros para visualizar dados que estejam acima ou abaixo de certos valores, permitindo uma análise customizada baseada em suas necessidades de trading.
- **Exportação de Dados:** Exporte uma tabela Resumo contendo informações detalhadas e anotações sobre ofertas e vendas realizadas para facilitar o processo de negociação.

## Começando

### Pré-requisitos

- Python 3.10
- Streamlit
- Pandas
- OpenPyXL

### Instalação

1. Clone o repositório do projeto:
   ```bash
      git clone https://github.com/rodrafanas/job_xls_trading_v013.git


2. Navegue até o diretório do projeto:
  ```bash
      cd job_xls_trading_v013
  ```

3. Instale as dependências necessárias:
  ```bash
      pip install -r requirements.txt
  ```
### Execução

Para rodar o dashboard, execute o seguinte comando no diretório do projeto:
  ```bash
      streamlit run streamlit_app.py
  ```
                      
### Uso

1. **Acesse o Dashboard:** Clique no link: [trading_dashbord](https://trading-app-vequis.streamlit.app/)
2. **Carregar Dados:** Inicie o dashboard e carregue os arquivos `.xlsx` correspondentes a cada lote de algodão.
3. **Definir Parâmetros:** Utilize os controles interativos para definir filtros baseados em Resistência da Fibra, Mic e UHM.
4. **Análise e Exportação:** Visualize os resultados na tabela interativa e utilize a opção de exportação para baixar a tabela Resumo com todas as informações relevantes.

## Contribuindo

Contribuições são sempre bem-vindas! Para contribuir, por favor:
1. Faça um Fork do projeto.
2. Crie uma Branch para sua Feature (`git checkout -b feature/AmazingFeature`).
3. Faça o Commit de suas mudanças (`git commit -m 'Add some AmazingFeature'`).
4. Faça o Push para a Branch (`git push origin feature/AmazingFeature`).
5. Abra um Pull Request.

## Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## Contato

Rafael Nascimento - [rodrafanas@gmail.com](mailto:rodrafanas@gmail.com)

Projeto Link: [https://github.com/rodrafanas/job_xls_trading_v013](https://github.com/rodrafanas/job_xls_trading_v013)
