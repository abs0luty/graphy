from collections.abc import Callable, Iterator
from typing import Optional, Self


class Edge:
    source: int
    destination: int
    weight: int

    def __init__(self, source: int, destination: int, weight: int = 1):
        assert source != destination, "Trying to create the edge from vertex to itself"
        
        self.source = source
        self.destination = destination
        self.weight = weight

    def __str__(self) -> str:
        return "Edge(%d, %d, %d)" % (self.source, self.destination, self.weight)

    def inverted(self) -> Self:
        return Edge(self.destination, self.source, self.weight)


class Child:
    vertex: int
    edge_weight: int

    def __init__(self, vertex: int, edge_weight: int):
        self.vertex = vertex
        self.edge_weight = edge_weight
    
    def __str__(self) -> str:
        return "Child(%d, %d)" % (self.vertex, self.edge_weight)


class Graph:
    children: list[list[Child]]
    vertices_count: int
    dijkstra_infinite_distance: int

    def __init__(self, vertices_count: int, dijkstra_infinite_distance: int = 1e7):
        self.children = []
        self.vertices_count = vertices_count
        self.dijkstra_infinite_distance = dijkstra_infinite_distance

        for _ in range(vertices_count):
            self.children.append([])

    def from_adjacency_matrix(self, matrix: list[list[int]]):
        """
        Constructors a graph frmo its adjacency matrix with weights
        """
        assert len(matrix) == self.vertices_count

        for i in range(len(matrix)):
            for j in range(len(matrix)):
                weight = matrix[i][j]

                if weight != 0:
                    self.add_edge(i, j, weight)

    def add_bidirectional_edge(self, edge: Edge):
        """
        Adds an edge between two given vertices in both directions.
        """
        self.add_edge(edge)
        self.add_edge(edge.inverted())

    def add_edge(self, edge: Edge):
        """
        Adds an edge between two given vertices.
        """
        assert edge.source < self.vertices_count, "Invalid vertex index"
        assert edge.destination < self.vertices_count, "Invalid vertex index"

        self.children[edge.source].append(Child(edge.destination, edge.weight))
    
    def children_of(self, vertex: int) -> Iterator[Child]:
        """
        Returns all children of the given vertex.

        .. highlight:: python
        .. code-block:: python
            graph = Graph(3)
            graph.add_edge(Edge(0, 2))
            graph.add_edge(Edge(1, 2, 2))

            assert graph.children_of(0)[0].vertex == 2
            assert graph.children_of(1)[0].vertex == 2
        """
        assert vertex < self.vertices_count, "Invalid vertex index"

        return self.children[vertex]
    
    def _dfs_through_edges(self, starting_vertex: int, 
            visited: list[int], 
            visit: Callable[[Edge], None]):
        """
        Single recursive iteration of DFS algorithm.
        """
        for child in self.children_of(starting_vertex):
            if not visited[child.vertex]:
                visited[child.vertex] = True
                visit(Edge(starting_vertex, child.vertex, child.edge_weight))
                self._dfs_through_edges(child.vertex, visited, visit)

    def dfs_through_edges(self, starting_vertex: int, 
            visited: Optional[list[int]] = None,
            visit: Callable[[Edge], None] = lambda _: None):
        """
        Traverse the graph using DFS algorithm.

        .. highlight:: python
        .. code-block:: python
            graph = Graph(3)
            graph.add_edge(Edge(0, 2))
            graph.add_edge(Edge(1, 2, 2))

            graph.dfs_through_edges(0, visit = lambda edge: print(edge.source))
            graph.dfs_through_edges(0, visit = lambda edge: print(edge.weight))
        """
        if visited == None:
            visited = [False] * self.vertices_count

        assert len(visited) == self.vertices_count

        visited[starting_vertex] = True
        self._dfs_through_edges(starting_vertex, visited, visit)
    
    def _dfs(self, starting_vertex: int, 
            visited: list[int], 
            visit: Callable[[Edge], None]):
        """
        Single recursive iteration of DFS algorithm.
        """
        for child in self.children_of(starting_vertex):
            if not visited[child.vertex]:
                visit(child.vertex)
                visited[child.vertex] = True
                self._dfs(child.vertex, visited, visit)

    def dfs(self, starting_vertex: int, 
            visited: Optional[list[int]] = None,
            visit: Callable[[int], None] = lambda _: None):
        """
        Traverse the graph using DFS algorithm.

        .. highlight:: python
        .. code-block:: python
            graph = Graph(3)
            graph.add_edge(Edge(0, 2))
            graph.add_edge(Edge(1, 2, 2))

            graph.dfs(0, visit = lambda x: print(x))
        """
        if visited == None:
            visited = [False] * self.vertices_count

        assert len(visited) == self.vertices_count

        if not visited[starting_vertex]:
            visit(starting_vertex)

        visited[starting_vertex] = True
        self._dfs(starting_vertex, visited, visit)

    def bfs_through_edges(self, starting_vertex: int,
            visited: Optional[list[int]] = None,
            visit: Callable[[Edge], None] = lambda _: None):
        """
        Traverse the graph using DFS algorithm.

        .. highlight:: python
        .. code-block:: python
            graph = Graph(3)
            graph.add_edge(Edge(0, 2))
            graph.add_edge(Edge(1, 2, 2))

            graph.bfs_through_edges(0, visit = lambda edge: print(edge.source))
            graph.bfs_through_edges(0, visit = lambda edge: print(edge.weight))
        """
        if visited == None:
            visited = [False] * self.vertices_count

        assert len(visited) == self.vertices_count

        visited[starting_vertex] = True
        queue = [starting_vertex]

        while len(queue) != 0:
            node = queue.pop(0)
            for child in self.children_of(node):
                if not visited[child.vertex]:
                    queue.append(child.vertex)
                    visit(Edge(node, child.vertex, child.edge_weight))
                    visited[child.vertex] = True

    def bfs(self, starting_vertex: int,
            visited: Optional[list[int]] = None,
            visit: Callable[[int], None] = lambda _: None):
        """
        Traverse the graph using DFS algorithm.

        .. highlight:: python
        .. code-block:: python
            graph = Graph(3)
            graph.add_edge(Edge(0, 2))
            graph.add_edge(Edge(1, 2, 2))

            graph.bfs(0, visit = lambda x: print(x))
        """
        if visited == None:
            visited = [False] * self.vertices_count

        assert len(visited) == self.vertices_count

        if not visited[starting_vertex]:
            visit(starting_vertex)

        visited[starting_vertex] = True
        queue = [starting_vertex]

        while len(queue) != 0:
            node = queue.pop(0)
            for child in self.children_of(node):
                if not visited[child.vertex]:
                    queue.append(child.vertex)
                    visit(child.vertex)
                    visited[child.vertex] = True

    def _minimum_distance_vertex(self, distances: list[int], visited: list[int]) -> int:
        """
        A utility function to find the vertex with
        minimum distance value, from the set of vertices
        not yet included in shortest path tree.
        """
        minimum_distance = 1e7
        minimum_distance_vertex = -1

        for vertex in range(self.vertices_count):
            if distances[vertex] < minimum_distance and not visited[vertex]:
                minimum_distance = distances[vertex]
                minimum_distance_vertex = vertex

        return minimum_distance_vertex

graph = Graph(7)
graph.add_bidirectional_edge(Edge(0, 1, 4))
graph.add_edge(Edge(1, 2, 3))
graph.add_edge(Edge(2, 3, 4))
graph.add_bidirectional_edge(Edge(3, 4))

graph.dfs_through_edges(0, visit = lambda x: print(x))
