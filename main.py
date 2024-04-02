import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Carregar os dados
nome_arquivo = 'teste5.npy'
arquivo = np.load(nome_arquivo)
x = arquivo[0]
y = np.ravel(arquivo[1])

# Definir as arquiteturas a serem testadas
arquiteturas = [
    (20,), 
    (15, 6),
    (10, 8, 5)
]

# Configurações para as simulações
num_simulacoes = 100
resultados = []

# Loop sobre as arquiteturas
for arquitetura in arquiteturas:
    melhores_resultados = []
    # Loop para executar cada simulação
    for _ in range(num_simulacoes):
        # Dividir os dados em conjunto de treinamento e teste
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # Ajuste o tamanho do conjunto de teste conforme necessário
        
        # Criar e treinar o modelo
        regr = MLPRegressor(hidden_layer_sizes=arquitetura,
                            max_iter=100,
                            activation='relu',
                            solver='adam',
                            learning_rate='adaptive',
                            n_iter_no_change=50)
        regr.fit(x_train, y_train)
        
        # Avaliar o modelo e calcular o erro final
        erro_final = np.mean((regr.predict(x_test) - y_test) ** 2)  # Por exemplo, erro quadrático médio
        melhores_resultados.append((erro_final, regr))
    
    # Selecionar o melhor resultado para esta arquitetura
    melhores_resultados.sort(key=lambda x: x[0])  # Ordenar pelos erros finais
    melhor_erro, melhor_modelo = melhores_resultados[0]
    
    # Adicionar o melhor resultado à lista de resultados
    resultados.append((arquitetura, melhor_erro, melhor_modelo))

# Imprimir os resultados
for resultado in resultados:
    print("Nome do arquivo:", nome_arquivo)
    print("Arquitetura:", resultado[0])
    print("Melhor erro final:", resultado[1])
    print()
    
    # Plotar o melhor resultado
    plt.figure(figsize=[14, 7])
    plt.plot(x, y, label="Original")
    plt.plot(x, resultado[2].predict(x), label="Predito")
    plt.title("Melhor resultado - Arquitetura {}".format(resultado[0]))
    plt.legend()
    plt.show()
