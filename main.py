from collections.abc import Callable, Iterable
from typing import Optional
from enum import Enum, auto


class TraverseAlgorithm(Enum):
    DFS = auto()
    BFS = auto()


class Edge:
    source: int
    destination: int
    weight: int

    def __init__(self, source: int, destination: int, weight: int = 1):
        assert source != destination, "Trying to create the edge from vertex to itself"
        
        self.source = source
        self.destination = destination
        self.weight = weight

    def __repr__(self) -> str:
        return "Edge(source=%d, dest=%d, weight=%d)" % (self.source, self.destination, self.weight)

    def inverted(self) -> 'Edge':
        return Edge(self.destination, self.source, self.weight)


class Child:
    vertex: int
    edge_weight: int

    def __init__(self, vertex: int, edge_weight: int):
        self.vertex = vertex
        self.edge_weight = edge_weight
    
    def __repr__(self) -> str:
        return "Child(vertex=%d, edge_weight=%d)" % (self.vertex, self.edge_weight)


class Graph:
    _children: list[list[Child]]
    _vertices_count: int

    def __init__(self, vertices_count: int):
        self._children = []
        self._vertices_count = vertices_count

        for _ in range(vertices_count):
            self._children.append([])

    @staticmethod
    def from_adjacency_matrix(matrix: list[list[int]]) -> 'Graph':
        """
        Constructors a graph frmo its adjacency matrix with weights
        """
        graph = Graph(len(matrix))

        for i in range(len(matrix)):
            for j in range(len(matrix)):
                weight = matrix[i][j]

                if weight != 0:
                    graph.add_edge(Edge(i, j, weight))

        return graph

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
        assert edge.source < self._vertices_count, "Invalid vertex index"
        assert edge.destination < self._vertices_count, "Invalid vertex index"

        child = Child(edge.destination, edge.weight)
        if child not in self._children[edge.source]:
            self._children[edge.source].append(child)

    def add_edges(self, edges: Iterable[Edge]):
        """
        Adds multiple edges to the graph.
        """
        for edge in edges:
            self.add_edge(edge)

    def add_bidirectional_edges(self, edges: Iterable[Edge]):
        """
        Adds multiple bidirectional edges to the graph.
        """
        for edge in edges:
            self.add_bidirectional_edge(edge)
    
    def children_of(self, vertex: int) -> Iterable[Child]:
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
        assert vertex < self._vertices_count, "Invalid vertex index"

        return self._children[vertex]
    
    def is_undirected(self) -> bool:
        """
        Returns True if the graph is directed, False otherwise.
        """
        is_undirected = True


        for source in range(self._vertices_count):
            for child in self._children[source]:
                if not(child in self._children[source] and 
                   source in self._children[child.vertex]):
                    is_undirected = False

        return is_undirected

    def to_undirected(self) -> 'Graph':
        """
        Converts the graph to an undirected graph.
        """
        undirected = Graph(self._vertices_count)

        for source in range(self._vertices_count):
            for child in self._children[source]:
                undirected.add_bidirectional_edge(Edge(source, child.vertex, child.edge_weight))
        
        return undirected
    
    def _dfs_through_edges_rec(self, 
                               starting_vertex: int, 
                               visited: list[bool],
                               visit: Callable[[Edge], None]):
        """
        Single recursive iteration of DFS algorithm.
        """
        for child in self.children_of(starting_vertex):
            if not visited[child.vertex]:
                visited[child.vertex] = True
                visit(Edge(starting_vertex, child.vertex, child.edge_weight))
                self._dfs_through_edges_rec(child.vertex, visited, visit)

    def _dfs_through_edges(self, 
                           starting_vertex: int, 
                           visited: Optional[list[bool]] = None,
                           visit: Callable[[Edge], None] = lambda _: None):
        """
        Traverse the graph using DFS algorithm.
        """
        if visited is None:
            visited = [False] * self._vertices_count

        assert len(visited) == self._vertices_count

        visited[starting_vertex] = True
        self._dfs_through_edges_rec(starting_vertex, visited, visit)
    
    def _dfs_rec(self, 
                 starting_vertex: int, 
                 visited: list[bool],
                 visit: Callable[[int], None]):
        """
        Single recursive iteration of DFS algorithm.
        """
        for child in self.children_of(starting_vertex):
            if not visited[child.vertex]:
                visit(child.vertex)
                visited[child.vertex] = True
                self._dfs_rec(child.vertex, visited, visit)

    def _dfs(self, 
             starting_vertex: int, 
             visited: Optional[list[bool]] = None,
             visit: Callable[[int], None] = lambda _: None):
        """
        Traverse the graph using DFS algorithm.
        """
        if visited is None:
            visited = [False] * self._vertices_count

        assert len(visited) == self._vertices_count

        if not visited[starting_vertex]:
            visit(starting_vertex)

        visited[starting_vertex] = True
        self._dfs_rec(starting_vertex, visited, visit)

    def _bfs_through_edges(self, 
                           starting_vertex: int,
                           visited: Optional[list[bool]] = None,
                           visit: Callable[[Edge], None] = lambda _: None):
        """
        Traverse the graph using BFS algorithm.
        """
        if visited is None:
            visited = [False] * self._vertices_count

        assert len(visited) == self._vertices_count

        visited[starting_vertex] = True
        queue = [starting_vertex]

        while len(queue) != 0:
            node = queue.pop(0)
            for child in self.children_of(node):
                if not visited[child.vertex]:
                    queue.append(child.vertex)
                    visit(Edge(node, child.vertex, child.edge_weight))
                    visited[child.vertex] = True

    def _bfs(self, 
             starting_vertex: int,
             visited: Optional[list[bool]] = None,
             visit: Callable[[int], None] = lambda _: None):
        """
        Traverse the graph using BFS algorithm.
        """
        if visited is None:
            visited = [False] * self._vertices_count

        assert len(visited) == self._vertices_count

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

    def traverse(self,
                 starting_vertex: int,
                 algorithm: TraverseAlgorithm = TraverseAlgorithm.BFS,
                 visited: Optional[list[bool]] = None,
                 visit: Callable[[int], None] = lambda _: None):
        """
        Traverses the graph using the given algorithm.
        """
        if algorithm == TraverseAlgorithm.BFS:
            self._bfs(starting_vertex, visited, visit)
        elif algorithm == TraverseAlgorithm.DFS:
            self._dfs(starting_vertex, visited, visit)

    def traverse_through_edges(self, 
                               starting_vertex: int,
                               algorithm: TraverseAlgorithm,
                               visited: Optional[list[bool]] = None,
                               visit: Callable[[Edge], None] = lambda _: None):
        """
        Traverses the graph using the given algorithm.
        """
        if algorithm == TraverseAlgorithm.BFS:
            self._bfs_through_edges(starting_vertex, visited, visit)
        elif algorithm == TraverseAlgorithm.DFS:
            self._dfs_through_edges(starting_vertex, visited, visit)

    def connected_components_count(self, 
                                   traverse_algorithm: TraverseAlgorithm = TraverseAlgorithm.BFS) -> int:
        """
        Returns the number of connected components in the graph.
        """
        visited = [False] * self._vertices_count
        count = 0

        for vertex in range(self._vertices_count):
            if not visited[vertex]:
                count += 1
                self.traverse(vertex, 
                              traverse_algorithm, 
                              visited, 
                              lambda x: None)

        return count
    
    def connected_components(self, 
                             traverse_algorithm: TraverseAlgorithm = TraverseAlgorithm.BFS) -> list[list[int]]:
        """
        Returns the connected components of the graph.
        """
        visited = [False] * self._vertices_count
        components = []

        for vertex in range(self._vertices_count):
            if not visited[vertex]:
                component = []
                self.traverse(vertex, 
                              traverse_algorithm, 
                              visited, 
                              lambda x: component.append(x))
                components.append(component)
        
        return components

            
    def _minimum_distance_vertex(self, 
                                 infinite_distance: int,
                                 distances: list[int], 
                                 visited: list[int]) -> int:
        """
        A utility function to find the vertex with
        minimum distance value, from the set of vertices
        not yet included in shortest path tree.
        """
        minimum_distance = infinite_distance
        minimum_distance_vertex = -1

        for vertex in range(self._vertices_count):
            if distances[vertex] < minimum_distance and not visited[vertex]:
                minimum_distance = distances[vertex]
                minimum_distance_vertex = vertex

        return minimum_distance_vertex
    
    def dijkstra(self, 
                 source: int,
                 infinite_distance: int = int(1e9)) -> list[int]:
        """
        Returns shortest distances from given source to all vertices in the graph.
        """
        distances = [infinite_distance] * self._vertices_count
        distances[source] = 0
        visited = [False] * self._vertices_count

        for _ in range(self._vertices_count):
            u = self._minimum_distance_vertex(infinite_distance, distances, visited)
            visited[u] = True

            for child in self.children_of(u):
                if not visited[child.vertex]:
                    new_distance = distances[u] + child.edge_weight
                    if new_distance < distances[child.vertex]:
                        distances[child.vertex] = new_distance

        return distances

graph = Graph(6)
graph.add_bidirectional_edges([
    Edge(1, 2),
    Edge(1, 3),
    Edge(3, 4),
    Edge(4, 5),
    Edge(2, 5)
])

print(graph.dijkstra(1, infinite_distance=int(1e11)))
