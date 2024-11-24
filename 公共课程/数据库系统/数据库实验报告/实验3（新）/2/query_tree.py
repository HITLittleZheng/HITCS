# query_tree.py

from graphviz import Digraph

class QueryNode:
    def __init__(self, operator, children=None, attributes=None, condition=None):
        self.operator = operator
        self.children = children if children else []
        self.attributes = attributes
        self.condition = condition
        self.id = id(self)  # 唯一标识符

    def __repr__(self):
        desc = self.operator
        if self.attributes:
            desc += f" [{', '.join(self.attributes)}]"
        if self.condition:
            desc += f" [Condition: {self.condition}]"
        return desc

    def to_graphviz(self, graph):
        label = self.__repr__()
        graph.node(str(self.id), label)
        for child in self.children:
            graph.edge(str(self.id), str(child.id))
            child.to_graphviz(graph)

    def visualize(self, filename='query_tree'):
        graph = Digraph(comment='Query Execution Tree')
        self.to_graphviz(graph)
        graph.render(filename, view=True)