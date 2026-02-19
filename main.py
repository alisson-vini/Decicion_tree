import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import decision_tree

data = load_iris(as_frame=True)
X = data.data
y = data.target

entrada_treino, entrada_teste, target_treino, target_teste = train_test_split(X, y, test_size=0.2, random_state=23)

# MINHA IMPLEMENTAÇÃO MANUAL
tree = decision_tree.Decision_tree()
tree.create_tree(entrada_treino, target_treino)
predict = pd.Series([ tree.predict(entrada_teste.iloc[i]) for i in range(len(entrada_teste)) ])

resultado_manual = (predict.values == target_teste.values).sum() / len(target_teste) * 100

# IMPLEMENTAÇÃO DO SCIKIT-LEARN 
tree = DecisionTreeClassifier()
tree.fit(entrada_treino, target_treino)
predict = tree.predict(entrada_teste)

resultado_sk = (predict == target_teste).sum() / len(target_teste) * 100


# COMPARAÇÃO ENTRE OS RESULTADOS
print(f"Resultado da implementação manual: {resultado_manual:.2f}")
print(f"Resultado da implementação sklearn: {resultado_sk:.2f}")
print(f"Diferença: {abs(resultado_sk - resultado_manual):.2f}")