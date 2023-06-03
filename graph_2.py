import numpy as np
import sys
from collections import deque

class Graph:
    def __init__(self, file_path, file_type):
        self.file_path = file_path
        self.file_type = file_type
        self.graph = self.load_graph()  # Загружаем граф

    def load_graph(self):
        if self.file_type == "-e":
            return self.load_list_of_edges()  # Загружаем список ребер
        elif self.file_type == "-m":
            return self.load_adjacency_matrix()  # Загружаем матрицу смежности
        elif self.file_type == "-l":
            return self.load_adjacency_list()  # Загружаем список смежности
        else:
            raise ValueError("Invalid file type")  # Некорректный тип файла

    def load_adjacency_list(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()  # Читаем строки файла
        num_vertices = len(lines)  # Определяем количество вершин
        matrix = np.zeros((num_vertices, num_vertices))  # Создаем матрицу смежности
        matrix[:] = np.inf  # Заполняем матрицу бесконечностями
        for i, line in enumerate(lines):  # Итерируемся по строкам
            neighbors = line.strip().split()  # Разделяем строки на элементы
            for neighbor in neighbors:  # Итерируемся по соседям
                matrix[i, int(neighbor) - 1] = 1  # Заполняем матрицу значениями
        return matrix

    def load_adjacency_matrix(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()  # Читаем строки файла
        num_vertices = len(lines)  # Определяем количество вершин
        matrix = np.zeros((num_vertices, num_vertices))  # Создаем матрицу смежности
        matrix[:] = np.inf  # Заполняем матрицу бесконечностями
        for i, line in enumerate(lines):  # Итерируемся по строкам
            row = line.strip().split()  # Разделяем строки на элементы
            for j, value in enumerate(row):  # Итерируемся по значениям строки
                matrix[i, j] = int(value) if int(value) != 0 else np.inf  # Заполняем матрицу значениями
        return matrix

    def adjacency_matrix(self):
        return self.graph  # Возвращаем матрицу смежности графа

    def load_list_of_edges(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()  # Читаем строки файла
        num_vertices = 0
        edges = []
        for line in lines:  # Итерируемся по строкам
            values = line.strip().split()  # Разделяем строки на элементы
            if len(values) == 2:  # Если количество элементов равно 2
                vertex1, vertex2 = values  # Присваиваем значения
                weight = 1
            else:
                vertex1, vertex2, weight = values  # Присваиваем значения
            edges.append((int(vertex1), int(vertex2), int(weight)))  # Добавляем ребро
            num_vertices = max(num_vertices, int(vertex1), int(vertex2))  # Обновляем количество вершин
        matrix = np.zeros((num_vertices, num_vertices))  # Создаем матрицу смежности
        matrix[:] = np.inf  # Заполняем матрицу бесконечностями
        for edge in edges:  # Итерируемся по ребрам
            vertex1, vertex2, weight = edge  # Присваиваем значения
            matrix[vertex1 - 1, vertex2 - 1] = weight  # Заполняем матрицу значениями
            matrix[vertex2 - 1, vertex1 - 1] = weight  # Заполняем матрицу значениями (для неориентированного графа)
        return matrix

    def list_of_edges(self, v):
        edges = []
        for i in range(self.graph.shape[0]):
            if self.graph[v - 1, i] != np.inf:
                edges.append((v, i + 1, self.graph[v - 1, i]))
        return edges

    def is_directed(self):
        return not np.array_equal(self.graph.transpose(), self.graph)  # Проверяем, является ли граф ориентированным

    def breadth_first_search(self, start_vertex):
        visited = np.zeros(self.graph.shape[0], dtype=bool)  # Создаем массив для отметки посещенных вершин
        queue = deque()  # Создаем очередь
        queue.append(start_vertex - 1)  # Добавляем начальную вершину в очередь
        visited[start_vertex - 1] = True  # Отмечаем начальную вершину как посещенную
        while queue:  # Пока очередь не пуста
            vertex = queue.popleft()  # Извлекаем вершину из очереди
            for neighbor in range(self.graph.shape[0]):  # Итерируемся по соседним вершинам
                if self.graph[vertex, neighbor] != np.inf and not visited[neighbor]:  # Если соседняя вершина не посещена
                    visited[neighbor] = True  # Отмечаем ее как посещенную
                    queue.append(neighbor)  # Добавляем ее в очередь
        return visited

    def connected_components(self):
        visited = np.zeros(self.graph.shape[0], dtype=bool)  # Создаем массив для отметки посещенных вершин
        components = []  # Создаем список компонент связности
        for vertex in range(1, self.graph.shape[0] + 1):  # Итерируемся по вершинам графа
            if not visited[vertex - 1]:  # Если вершина не посещена
                component = self.breadth_first_search(vertex)  # Выполняем поиск в ширину
                components.append(component)  # Добавляем компоненту связности в список
                visited = np.logical_or(visited, component)  # Обновляем массив посещенных вершин
        return components

    def strong_connectivity(self):
        index_counter = [0]
        stack = []
        lowlinks = np.zeros(self.graph.shape[0], dtype=int)
        index = np.full(self.graph.shape[0], -1, dtype=int)
        on_stack = np.zeros(self.graph.shape[0], dtype=bool)
        components = []

        def strongconnect(vertex):
            index[vertex] = index_counter[0]
            lowlinks[vertex] = index_counter[0]
            index_counter[0] += 1
            stack.append(vertex)
            on_stack[vertex] = True

            for neighbor in range(self.graph.shape[0]):
                if self.graph[vertex, neighbor] != np.inf:
                    if index[neighbor] == -1:
                        strongconnect(neighbor)
                        lowlinks[vertex] = min(lowlinks[vertex], lowlinks[neighbor])
                    elif on_stack[neighbor]:
                        lowlinks[vertex] = min(lowlinks[vertex], index[neighbor])

            if lowlinks[vertex] == index[vertex]:
                component = np.zeros(self.graph.shape[0], dtype=bool)
                while True:
                    v = stack.pop()
                    on_stack[v] = False
                    component[v] = True
                    if v == vertex:
                        break
                components.append(component)

        for vertex in range(self.graph.shape[0]):
            if index[vertex] == -1:
                strongconnect(vertex)

        return components

print("Введите ключ параметра:")
print("-e: list_of edges, \n-m: matrix, \n-l: list_of_adjacency")
key = input()
if key not in ['-m', '-e', '-l']:
    print('Неверный тип ключа!')
    sys.exit()
print("Введите название файла (в текущем каталоге):")
file = input()
print('\n')

g = Graph(file, key)

adj_matrix = g.adjacency_matrix()
np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=8, suppress=True)

# Запись результатов в файл
with open("output.txt", 'w') as file:
    file.write("Adjacency matrix:\n")
    for row in adj_matrix:
        row_str = ' '.join(map(str, row))
        file.write(row_str + '\n')
    file.write('\n')

# Проверка связности
if g.is_directed():
    print("Digraph is connected.")
    components = g.strong_connectivity()
    num_components = len(components)
    print("Connected components:")
    for i, component in enumerate(components):
        vertices = np.nonzero(component)[0] + 1
        print(vertices)
    print("Digraph is weakly connected and contains", num_components, "strongly connected components.")
    print("Strongly connected components:")
    for i, component in enumerate(components):
        vertices = np.nonzero(component)[0] + 1
        print(vertices)
else:
    print("Graph is connected.")
    components = g.connected_components()
    num_components = len(components)
    print("Connected components:")
    for i, component in enumerate(components):
        vertices = np.nonzero(component)[0] + 1
        print(vertices)
    print("Graph is weakly connected and contains", num_components, "connected components.\n")


print("Результаты записаны в файл output.txt")