"""
Projeto Ciência de Dados - Previsão de Vendas
 
O desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa X investe: TV, Jornal e Rádio

Passo a Passo
 
- Passo 1: Entendimento do Desafio
- Passo 2: Entendimento da Área/Empresa
- Passo 3: Extração/Obtenção de Dados
- Passo 4: Ajuste de Dados (Tratamento/Limpeza)
- Passo 5: Análise Exploratória
- Passo 6: Modelagem + Algoritmos
- Passo 7: Interpretação de Resultados

"""

import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

dados = pd.read_csv('advertising.csv')

display(dados)
print(dados.info())

sns.heatmap(dados.corr(), annot = True, cmap = 'Blues')
plt.show()

sns.pairplot(dados, corner=True, height=1.5)
plt.show()

y = dados['Vendas']
X = dados.drop(columns= ['Vendas'])

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y)

# cria as inteligencias aritificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina as inteligencias artificias
modelo_regressaolinear.fit(X_treino, y_treino)
modelo_arvoredecisao.fit(X_treino, y_treino)

# Criar as previsões
previsao_regressaolinear = modelo_regressaolinear.predict(X_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(X_teste)

# Comparar os modelos
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao)) 

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()

sns.barplot(x=X_treino.columns, y=modelo_arvoredecisao.feature_importances_, color="deepskyblue")
plt.show()

# Salvando tabelas auxiliares para fazer novos gráficos
tabela_auxiliar.to_excel('Tabela Auxiliar do Machine Learning.xlsx')

tabela_auxiliar2 = pd.DataFrame()
tabela_auxiliar2['Canal'] = X_treino.columns
tabela_auxiliar2['Nível de Importância'] = modelo_arvoredecisao.feature_importances_
tabela_auxiliar2.to_excel('Tabela Auxiliar de Importância.xlsx')