import pandas as pd
import numpy as np

# classe do Node que vai ser o elemento básico da arvore
class Node():
    def __init__(self, nome_coluna=None, threshold=None):
        self.nome_coluna = nome_coluna
        self.threshold = threshold
        self.left = None               # outro node para valores <= threshold
        self.right = None              # outro node para valores >  threshold
        self.value = None              # if != None, is the predict
        self.value_proba = {}          # dicionario para armazenar a quantidade de aparições de cada valor único do target da tabela atual do nó

class Decision_tree():
    def __init__(self, max_deep=None, min_samples_split:int=2, min_samples_leaf:int=1, min_impurity_decrease:float=0.0):
        self.no_raiz = Node()                              # Node principal de onde a arvore começa
        #self.true_deep = 0                                 # profundidade real da arvore
        self.max_deep = max_deep                           # profundidade máxima da arvore
        self.min_samples_split = min_samples_split         # número mínimo de amostras para fazer um split
        self.min_samples_leaf = min_samples_leaf           # quantidade mínima de elementos da folha para ela existir
        self.min_impurity_decrease = min_impurity_decrease # número mínimo de redução de pureza para poder gerar um split

    # HELP FUNCTION
    def creat_leaf(self, current_node:Node):
        current_node.value = max(current_node.value_proba, key=current_node.value_proba.get) # coloca como valor o elemento que mais aparece

    def tratar_tabela_split(self, table:pd.DataFrame, column:str, target:pd.Series, threshold:float, direction:str="left"):
        """
            Função para tratar a tabela antes dela ir para o próximo node, são feitas as seguintes ações:
                - remover todas as linhas que não passam no filtro do threshold
                    "left" = linhas onde valor <= Threshold
                    "right" = linhas onde o valor > Threshold
                - remover a coluna que acabou de ser escolhida (a mesma coluna não pode ser escolhida duas vezes)
                
            parâmetros:
                tabela -> um DataFrame que é a tabela onde ele vai dazer a limpeza
                column -> é a coluna que vai ser removida (ultima utilizada)
                direction -> uma string entre ("left", "right"), explicado posteriormente quando usar cada uma delas

            return:
                um novo DataFreme com os filtros a cima
                uma série do target com os filtros das linhas
        """

        new_table = table.copy()            # cria uma cópia da tabela para evitar modificações na original
        
        if    direction == "left": mask = new_table[column] <= threshold
        elif  direction == "right": mask = new_table[column] > threshold
        else: return # adicionar código para erro

        new_table  = new_table[mask]        # remove as linhas que não atedem as condições (dependendo de left/right)
        #new_table  = new_table.drop(column) # remove a coluna
        new_target = target[mask]           # pega a coluna de target com os filtros das linhas 

        return (new_table, new_target)

    # função para achar o melhor Threshold dado um um array e seus targets
    def threshold_gini(self, array_values, array_target):
        """
            Gera qual o melhor threhold dado um conjunto de vários valores numéricos e os seus repectivos targets

            array_values -> é o array com todos os valores
            array_target -> é o array que contem todos os targets para medir a Gini/Entropia e classificar quem é melhor

            return -> uma tupla com os seguintes valores:
            threshold:float
            gini do melhor threshold:float

            alusão fisual:

            
        """

        array_target = np.asanyarray(array_target)
        array_values = np.asanyarray(array_values)

        # Armazena todos os Thresholds com ruas respectivas purezas para escolher o melhor
        threshold = 0.0     # threshold atual
        best_gini = float("inf") # menor geni atual
        
        values = np.asanyarray(array_values) # passa para o tipo array NumPy
        values = np.unique(values)           # pega todos os valores únicos do array
        values = np.sort(values)             # coloca os valores em ordem crescente

        # para o caso do array so ter um valor único (gera um split com uma amostra vazia)
        if len(values) < 2: return (None, None)

        # Vai interar sobre todos os pares consecutivos gerados por (i,i+1), (i+1, i+2)... (ix-1, ix)
        for i in range(len(values) - 1):
            threshold_atual = (values[i] + values[i+1]) / 2   # calcula o Threshold para esse par de valores

            # calcula qual a Geni dado esse Threshold
            mascara = array_values <= threshold_atual

            target_1 = array_target[mascara]  # pega todos os targets onde os valores <= Threshold (targets a esquerda)
            gini_1 = gini_calculate(target_1)

            target_2 = array_target[~mascara] # pega todos os targets onde os valores <= Threshold (target a direita)
            gini_2 = gini_calculate(target_2)

            # ganho de informação gini = (gini_1 * peso_1 + gini_2 * peso_2) / peso_1 + peso_2
            total_peso = len(array_target)
            gini_ponderado = ((gini_1 * len(target_1)) + (gini_2 * len(target_2))) / total_peso

            if gini_ponderado < best_gini:
                best_gini = gini_ponderado
                threshold = threshold_atual
        
        return (threshold, best_gini)

    # seleciona a coluna que tem maior ganho de informação dentro da tabela e retorna: (column_name, threshold, gini,)
    def selecionar_coluna(self, table:pd.DataFrame, target):
        """
        
        """
        
        threshold = int()        # armazena o melhor threshold da coluna com menor gini
        best_gini = float("inf") # armazena o menor gini dentre as colunas
        column_name = str()

        # percorre todas as colunas
        for coluna in table.columns:
            threshold_temp, gini_temp = self.threshold_gini(table[coluna], target)

            if threshold_temp == None: continue # Para o caso de ter uma coluna somente com valores únicos que vai gerar um split vazio

            # se o gini atual for melhor do que o antigo nos salvamos as informações dessa coluna ()
            if gini_temp < best_gini:
                threshold = threshold_temp
                best_gini = gini_temp
                column_name = coluna

        if best_gini == float("inf"): return None, None, None # para o caso de todas as colunas serem invalidas por terem valores unicos
        return(column_name, threshold, best_gini)

    

    def create_tree(self, table:pd.DataFrame, target:pd.Series, current_node:Node=None, deep:int=1):
        """
        
        """

        if current_node == None: current_node = self.no_raiz             # caso o nó esteja vazio (só é feito uma vez no começo da função)
            
        valores_unicos, contagem = np.unique(target, return_counts=True) # pega os valores únicos e suas quantidades
        current_node.value_proba = {valor:taxa_aparicao for valor, taxa_aparicao in zip(valores_unicos, contagem)} # gera um dict com os valores únicos como chave e sua taxa de aparição como o valor

        column_name, threshold, gini = self.selecionar_coluna(table, target)  # Procura a melhor coluna, obtem seu threshold e gini
        # caso todas as colunas forem invalidas transforma o no em folha
        if column_name is None or threshold is None or gini is None:
            self.creat_leaf(current_node)
            return

        father_impurity = gini_calculate(target)   # usado para calcular a redução de impureza
        impurity_reduction = father_impurity - gini # redução de impureza usado nos critérios de parada

        # CRITÉRIOS DE PARADA
         # quando o conjunto de targets é puro
        if len(valores_unicos) == 1:
            current_node.value = target.iloc[0] # primeiro valor (que é igual a todos os valores)
            return
        
        # se a quantidade de amostras for pequena demais
        elif len(target) < self.min_samples_split:
            self.creat_leaf(current_node)
            return
        
        # se a arvore atingir a profundidade máxima permitida
        elif self.max_deep is not None and self.max_deep <= deep:
            self.creat_leaf(current_node)
            return
        
        # Quando a redução de impureza é muito pequeno
        elif impurity_reduction < self.min_impurity_decrease:
            self.creat_leaf(current_node)
            return

        # caso em que um dos nós não tem nenhum valor (isso não deve acontecer pois as funções foram feitas para evitar isso)
        elif len(target) == 0:
            raise ValueError(f"Nó sem amostras :(")


        # CONSOLIDANDO O NÓ
        # Preeche o nó com o nome da coluna escolhida e seu threshold
        current_node.nome_coluna = column_name
        current_node.threshold = threshold


        # SPLITS
        # Prepara os dados para o Node a esquerda (valor <= Threshold) e chama novamente a função para preenche-lo
        new_table_left, new_target_left = self.tratar_tabela_split(table, column_name, target, threshold, "left")    # remove as linhas onde os valores > threshold e a coluna selecionada a cima
        # Prepara os dados para o Node a direita (valor > Threshold) e chama novamente a função para preenche-lo
        new_table_right, new_target_right = self.tratar_tabela_split(table, column_name, target, threshold, "right") # remove as linhas ond os valores > threshold e a coluna selecionada a cima

        # caso os filhos gerados tenham poucas amostras (criterio de parada)
        if len(new_target_left) < self.min_samples_leaf or len(new_target_right) < self.min_samples_leaf:
            self.creat_leaf(current_node)
            return

        new_node = Node()           # cria um novo node
        current_node.left = new_node # liga node filho da esquerda ao nó pai

        self.create_tree(new_table_left, new_target_left, new_node, deep+1)

        new_node = Node()            # cria um novo node
        current_node.right = new_node # liga node filho da direita ao nó pai

        self.create_tree(new_table_right, new_target_right, new_node, deep+1)

    def predict(self, row:pd.Series, current_node:Node=None, nome_coluna:str="target"):
        """
        
        """

        if current_node is None: current_node = self.no_raiz # para o caso de ser o primeiro nó

        # Critério de parada
        if current_node.value != None:
            return current_node.value
        
        # percorre para a esquerda (valor <= Threshold)
        if row[current_node.nome_coluna] <= current_node.threshold:
            return self.predict(row, current_node.left)
        
        # percorre para a direita (valor > Threshold)
        else:
            return self.predict(row, current_node.right)
        

# função para calcular a Gini de um array
def gini_calculate(array):
    """
        Função para calcular a Gini dado um determinado array, ou seja vai medir quão bem distribuido,
        esses arrays são gerados pelo split do filtro de uma coluna
    """

    array = np.asarray(array) # passa para o tipo array NumPy

    # array com a lista de i probabilidades calculadas com base nos diferente elementos do array
    _, contagem = np.unique(array, return_counts=True)
    array_prob = contagem / len(array)

    # retorna o valor da Gini
    return 1 - np.sum(array_prob**2)
