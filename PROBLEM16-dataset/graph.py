from collections import defaultdict
class Node:
    def __init__(self, dicID, dic, dicComponent, component, color, root=False):
        self.dic = dic
        self.dicComponent = dicComponent
        self.id = dicID
        self.history = [component, color]
        
        if root:
            self.root = True
            self.data = [self.id, 0, root]
        else:
            self.root = root
            self.data = [self.id, 1, root]

        self.children = []
        self.parent = None
        self.readableChildren = []
        self.g = 0
        self.h = 0
        self.f = 0
        self.level = 0  

    def __lt__(self, other):
        return self.f < other.f
        
        
    def add_child(self, node):
        node.parent = self
        if node not in self.children:
            self.children.append(node)
        if node.data not in self.readableChildren:
            self.readableChildren.append(node.data)


