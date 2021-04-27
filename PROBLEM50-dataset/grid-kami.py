import os
import csv
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
from export_to_csv import convert_from_txt_to_csv
from collections import defaultdict
from itertools import combinations
from graph import Node
import networkx as nx
from timeout import timeout
import time

class Counter:
    def __init__(self):
        self.iteration = 0


class grid:
    def __init__(self, problem_file_name, optimization):
        self.duplicate_dictionary = {}
        self.duplicate_list = []
        self.shortest_length = 0
        self.i = 0
        self.iterationcount = 0
        self.problem_file_name = problem_file_name
        self.componentsList = []
        self.counter = 0
        self.totalBoxes = 0
        self.neighbor_dict = dict()
        #get size of grid
        total_lines = 0
        total_characters = 0
        with open(self.problem_file_name, 'r') as f:
            for line in f:
                total_characters = len(line.strip())
                total_lines += 1

        # Make a grid...
        self.nrows = total_lines
        self.ncols = total_characters
        #self.image = [[[] for i in range(self.ncols)] for j in range(self.nrows)]
        self.image = np.zeros([self.nrows, self.ncols])
        count_lines = 0
        count_characters = 0

        # Set every cell to a number (this would be your data)
        with open(self.problem_file_name, 'r') as f:
            for line in f:
                #print("line:", count_lines, line)
                for letter in line.strip():
                    #print("character: ", count_characters, letter)
                    if letter == 'B':
                        self.image[count_lines][count_characters] = 0
                    elif letter == 'Y':
                        self.image[count_lines][count_characters] = 1
                    elif letter == 'R':
                        self.image[count_lines][count_characters] = 0.6
                    
                    count_characters = (count_characters + 1) % (total_characters)
                count_lines = (count_lines + 1) % (total_lines)
        
        self.initNeighborGraph()
        self.new_image = copy.deepcopy(self.image)

    def reset(self):
        self.image = copy.deepcopy(self.new_image)
        self.totalBoxes = 0
                
    def plot(self):
        row_labels = range(self.nrows)
        col_labels = range(self.ncols)
        plt.matshow(self.image)
        plt.xticks(range(self.ncols), col_labels)
        plt.yticks(range(self.nrows), row_labels)
        #plt.draw()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close("all")


    def getGrid(self):
        return self.image

    def initNeighborGraph(self):
        for row in range(len(self.image)):
            for col in range(len(self.image[0])):
                tile = self.image[row][col]
                tile_name = str(row) + "x" + str(col)
                self.neighbor_dict[tile_name] = list()

                #right
                right = col + 1;
                right_tile_name = str(row) + "x" + str(right)
                if right < len(self.image[0]):
                    right_tile = self.image[row][right];
                    if right_tile_name not in self.neighbor_dict[tile_name]:
                        self.neighbor_dict[tile_name].append(right_tile_name)
                #down
                down = row + 1;
                down_tile_name = str(down) + "x" + str(col)
                if down < len(self.image):
                    down_tile = self.image[down][col];
                    if down_tile_name not in self.neighbor_dict[tile_name]:
                        self.neighbor_dict[tile_name].append(down_tile_name)
                #left
                left = col - 1;
                left_tile_name = str(row) + "x" + str(left)
                if left >=0:
                    left_tile = self.image[row][left];
                    if left_tile_name not in self.neighbor_dict[tile_name]:
                        self.neighbor_dict[tile_name].append(left_tile_name)
                #up
                up = row - 1;
                up_tile_name = str(up) + "x" + str(col)
                if up >= 0:
                    temp_UpTile = self.image[up][col];
                    if up_tile_name not in self.neighbor_dict[tile_name]:
                        self.neighbor_dict[tile_name].append(up_tile_name)

    def getNeighborGraph(self, key=None):
        if key is None:
            return self.neighbor_dict
        else:
            return self.neighbor_dict[key]

    def change_color(self, newColor, row, col):
        print("change : ", str(row) + "x" + str(col), " to ", newColor)
        #self.totalBoxes = 1
        oldColor = self.image[row][col]
        if oldColor == newColor:
            return

        reachableNodes = []
        listButton = self.getNeighborGraph(str(row) + "x" + str(col))
        self.image[row][col] = newColor


        while (True):
            for button in listButton:
                row_, col_ = button.split("x")
                row_ = int(row_)
                col_ = int(col_)
                if self.image[row_][col_] == oldColor:
                
                    if button not in reachableNodes:
                        reachableNodes.append(button)
                        self.image[row_][col_] = newColor
            
            if len(reachableNodes) < 1:
                break
            
            listButton = []
            listButton = self.getNeighborGraph(reachableNodes[0])
            del reachableNodes[0];
            if len(listButton) < 1:
                break;
        
    def checkConnectedColors(self, image, x, y):
        self.counter += 1
        visited = []
        queue = []
        row = x
        col = y
        color = image[row][col]

        visited.append(str(row) + "x" + str(col))
        queue.append(str(row) + "x" + str(col))
        image[row][col] = -1
        self.totalBoxes = 1

        while (queue):
            s = queue.pop(0)
            for button in self.getNeighborGraph(s):
                row_, col_ = button.split("x")
                row_ = int(row_)
                col_ = int(col_)
                
                if button not in visited and image[row_][col_] == color:
                    visited.append(button)
                    queue.append(button)
                    #print("appending:   ", button)
                    self.totalBoxes += 1
                    image[row_][col_] = -1
                    
        self.componentsList.append([visited, color])
        #print("componentslist: ", len(self.componentsList), self.componentsList)

    def getComponents(self):
        image = copy.deepcopy(self.image)
        self.componentsList = []
        self.initNeighborGraph()
        for row in range(self.nrows):
            for col in range(self.ncols):
                if image[row][col] != -1:
                    self.checkConnectedColors(image, row, col)
        #print(self.getGrid())
        #self.reset()
        return self.componentsList

    def getConComponents(self):
        
        dic = defaultdict(list)
        #dicColor = dict()
        dicComponent = dict()
        lst = self.getComponents()
        for combo1, combo2 in combinations(lst,2):
            
            component1, color1 = combo1
            component2, color2 = combo2
            index1 = lst.index([component1, color1])
            index2 = lst.index([component2, color2])
            dicComponent[index1] = [component1, color1]
            dicComponent[index2] = [component2, color2]
            
            #print(index1, index2)
            for button in component1:
                row_, col_ = button.split("x")
                row_ = int(row_)
                col_ = int(col_)
                listNeighborButton = self.getNeighborGraph(str(row_) + "x" + str(col_))
                #print(tilei, "\n", listButtoni)
                for i in listNeighborButton:
                    if i in component2:
                        dic[index1].append(index2)
                        dic[index1] = list(set(dic[index1]))
                        dic[index2].append(index1)
                        dic[index2] = list(set(dic[index2]))
                        #dicColor[index1] = color1
                        #dicColor[index2] = color2             
            
            
        #print(dic, "\n", dicComponent)
        return [dic, dicComponent]





        
    def changeColorOfComponentGraph(self, dictionary, dicComp, component, color):
        #dic, dicComponent = self.getConComponents()
        dic = dictionary
        dicComponent = dicComp
        dicComponent[component][1] = color
        deleteList = []
        connectedNodes = []
        
        for key in dic.keys():
            neighbors = dic[key]
            for neighbor in neighbors:
                print(dicComponent[key][1], dicComponent[neighbor][1])
                if dicComponent[key][1] == dicComponent[neighbor][1]:
                    if neighbor not in deleteList:
                        deleteList.append(neighbor)
                    if key not in deleteList:
                        deleteList.append(key)
                        
        print("delete: ", deleteList)
        
        """Check if neighboring Nodes have same color. Therefore deleteList should not be empty
        """
        if len(deleteList) > 1:
            for key in deleteList:
                print(key)
                for neighbor in dic[key]:
                    print(neighbor)
                    if neighbor not in deleteList:
                        connectedNodes.append(neighbor)
            
                    
            connectedNodes = list(set(connectedNodes))
            
            print("connected before eliminating: ", connectedNodes)
            connectedNodes1 = connectedNodes[:]


            
            for elem in connectedNodes:
                if elem in deleteList:
                    print("True")
                    connectedNodes1.remove(elem)

            newKey = deleteList[0]
            connectedNodes = connectedNodes1
            #print(dic)
            newDic = copy.deepcopy(dic)
            newDicComponent = copy.deepcopy(dicComponent)
            for key in dic.keys():
                print("key:", key)
                for neighbor in dic[key]:
                    print("neigh:", neighbor)
                    if neighbor in deleteList:
                        print("delete:", neighbor)
                        #newDic.pop(neighbor, None)
                        newDic[key].remove(neighbor)
                        
                if key in deleteList:
                    del newDic[key]
                    if key != newKey:
                        del newDicComponent[key]
            print("connected after eliminating: ", connectedNodes)
            #print("final dic", newDic)
            #print(dic)


            """Add transitions for new merged Key 
            """
            
            mergeComponent = dicComponent[newKey][0]
            print("newKey: ", newKey, "\nmerged component: ", mergeComponent)
            newDic[newKey] = connectedNodes
            for key in newDic.keys():
                if key in connectedNodes:
                    newDic[key].append(newKey)
                    newDicComponent[newKey][1] = color
                    newDicComponent[newKey][0] = mergeComponent
                    

            return newDic, newDicComponent
        else:
            return dic, dicComponent

        
        
    def expand_node(self, node):
        dic = copy.deepcopy(node.dic)
        dicComponent = copy.deepcopy(node.dicComponent)
        self.iterationcount = 0
        if optimization in ["dd1", "ssr-dd1"]:
            matrix = create_matrix_for_DD(dic, dicComponent)
            self.duplicate_dictionary[matrix.tostring()] = True
            self.duplicate_dictionary[mirror_matrix(matrix).tostring()] = True
        elif optimization in ["dd2", "dd3"]:
            graph = from_dic_to_networkx_graph(dic, dicComponent)
            self.duplicate_list.append(graph)

        if len(dic.keys()) == 1:
            print("GAME OVER!!!")
            return True
        else:
            lst = []
            solutionLst = []
            
            for component in dic.keys():
                for color in [0.0, 0.6, 1.0]:
                    """optimization: only look at changed states
                    """
                    print("check:  ", "component:", component, "componentColor:", copy.deepcopy(node.dicComponent)[component][1], "color:", color)
                    
                    if copy.deepcopy(node.dicComponent)[component][1] != color:
                        
                        if optimization == "normal":
                            self.iterationcount += 1
                            dic = copy.deepcopy(node.dic)
                            dicComponent = copy.deepcopy(node.dicComponent)
                            #print(dic, dicComponent)
                            newDic, newDicComponent = self.changeColorOfComponentGraph(dic, dicComponent, component, color)
                            lst.append([newDic, newDicComponent, dicComponent[component][0], color, self.iterationcount])
                        elif optimization == "dd1":
                            dic = copy.deepcopy(node.dic)
                            dicComponent = copy.deepcopy(node.dicComponent)
                            #print(dic, dicComponent)
                            newDic, newDicComponent = self.changeColorOfComponentGraph(dic, dicComponent, component, color)
                            newMatrix = create_matrix_for_DD(newDic, newDicComponent)
                            if newMatrix.tostring() not in self.duplicate_dictionary:
                                self.iterationcount += 1
                                print("adding node that isnt in duplicate memory...\n", newMatrix)
                                self.duplicate_dictionary[newMatrix.tostring()] = True
                                self.duplicate_dictionary[mirror_matrix(newMatrix).tostring()] = True
                                lst.append([newDic, newDicComponent, dicComponent[component][0], color, self.iterationcount])
                        elif optimization == "ssr":
                            dic = copy.deepcopy(node.dic)
                            dicComponent = copy.deepcopy(node.dicComponent)
                            newDic, newDicComponent = self.changeColorOfComponentGraph(dic, dicComponent, component, color)
                            if len(newDic) < len(dic):
                                self.iterationcount += 1
                                lst.append([newDic, newDicComponent, dicComponent[component][0], color, self.iterationcount])
                        elif optimization == "ssr-dd1":
                            dic = copy.deepcopy(node.dic)
                            dicComponent = copy.deepcopy(node.dicComponent)
                            #print(dic, dicComponent)
                            newDic, newDicComponent = self.changeColorOfComponentGraph(dic, dicComponent, component, color)
                            newMatrix = create_matrix_for_DD(newDic, newDicComponent)
                            if newMatrix.tostring() not in self.duplicate_dictionary and len(newDic) < len(dic):
                                self.iterationcount += 1
                                print("adding node that isnt in duplicate memory...\n", newMatrix)
                                self.duplicate_dictionary[newMatrix.tostring()] = True
                                self.duplicate_dictionary[mirror_matrix(newMatrix).tostring()] = True
                                lst.append([newDic, newDicComponent, dicComponent[component][0], color, self.iterationcount])
                        elif optimization == "dd2":
                            dic = copy.deepcopy(node.dic)
                            dicComponent = copy.deepcopy(node.dicComponent)
                            newDic, newDicComponent = self.changeColorOfComponentGraph(dic, dicComponent, component, color)
                            newGraph = from_dic_to_networkx_graph(newDic, newDicComponent)
                            isomorphic = False
                            for graph in self.duplicate_list:
                                if nx.is_isomorphic(newGraph, graph, node_match=colors_match):
                                    isomorphic = True
                                    break
                            if not isomorphic:
                                self.iterationcount += 1
                                print("adding node that isnt in duplicate memory...\n")
                                self.duplicate_list.append(newGraph)
                             
                                lst.append([newDic, newDicComponent, dicComponent[component][0], color, self.iterationcount])
                        elif optimization == "dd3":
                            dic = copy.deepcopy(node.dic)
                            dicComponent = copy.deepcopy(node.dicComponent)
                            newDic, newDicComponent = self.changeColorOfComponentGraph(dic, dicComponent, component, color)
                            newGraph = from_dic_to_networkx_graph(newDic, newDicComponent)
                            isomorphic = False
                            for graph in self.duplicate_list:
                                if nx.faster_could_be_isomorphic(newGraph, graph):
                                    if nx.is_isomorphic(newGraph, graph, node_match=colors_match):
                                        isomorphic = True
                                        break
                            if not isomorphic:
                                self.iterationcount += 1
                                print("adding node that isnt in duplicate memory...\n")
                                self.duplicate_list.append(newGraph)
                             
                                lst.append([newDic, newDicComponent, dicComponent[component][0], color, self.iterationcount])


            nodeList = []
            for i in lst:
                newNodeID = node.id + "." + str(i[4])
                newNode = Node(newNodeID, i[0], i[1], i[2][0], i[3], False)
                nodeList.append(newNode)

            return nodeList

def create_matrix_for_DD(dic, dicComponent):
    colorsA = dicComponent
    matrixA = np.zeros([len(dic.keys()), len(dic.values())])
    
    
    for i,k in enumerate(dic.keys()):
        for j, kk in enumerate(dic.values()):
            if i != j:
                matrixA[i][j] = colorsA[k][1]
            else:
                matrixA[i][j] = -1
    return matrixA

def mirror_matrix(matrix):
    return np.flip(np.flip(matrix, 0), 1)

def from_dic_to_networkx_graph(dic, dicColor):

    G = nx.DiGraph(dic)
    for n in G.nodes():
        if dicColor[n][1] == 0.0:
            G.nodes[n]['color'] = 'b'
        elif dicColor[n][1] == 0.6:
            G.nodes[n]['color'] = 'r'
        elif dicColor[n][1] == 1.0:
            G.nodes[n]['color'] = 'y'

    return G


#https://stackoverflow.com/questions/32363592/colored-graph-isomorphism
def colors_match(n1_attrib,n2_attrib):
    '''returns False if either does not have a color or if the colors do not match'''
    try:
        return n1_attrib['color']==n2_attrib['color']
    except KeyError:
        return False


def hComponents(dic, dicComponent):

        minKeyList = []

        """ Getting node that can eliminate a whole color
        """
        print("DOING 1st STRATEGY")
        totalBlue = 0
        totalRed = 0
        totalYellow = 0
        try1stWay = False
        #get number of color appearances in dictionary
        for key in dic.keys():
            color = dicComponent[key][1]
            if color == 0.0:
                totalBlue += 1
            elif color == 0.6:
                totalRed += 1
            elif color == 1.0:
                totalYellow += 1
        print("total colors: ", totalBlue, totalRed, totalYellow)
        #get total colors in dictionary
        totalColors = 0
        for color in [totalBlue, totalRed, totalYellow]:
            if color != 0:
                totalColors += 1
        print("total: ", totalColors)
        #check if key can eliminate a whole color at once
        for key in dic.keys():
            print("key: ", key)
            neighborsList = dic[key]
            print("key neighborslist: ", neighborsList)
            blueSum = 0
            redSum = 0
            yellowSum = 0
            #if key does not only have 1 neighbor. ==> if key is not a dead end.
            if len(neighborsList) > 1:
                for neighbor in neighborsList:
                    color = dicComponent[neighbor][1]
                    if color == 0.0:
                        blueSum += 1
                    elif color == 0.6:
                        redSum += 1
                    elif color == 1.0:
                        yellowSum += 1
                
                print("not dead end! ", blueSum, redSum, yellowSum)
                if blueSum == totalBlue and totalBlue != 0:
                    return totalColors - 1
                if redSum == totalRed and totalRed != 0:
                    print("appending red key: ", key)
                    return totalColors - 1
                if yellowSum == totalYellow and totalYellow != 0:
                    return totalColors - 1

        return len(dic) - 1
    

 


def hColors(dic, dicComponent):
    components = int(len(dic) - 1)
    #return components
    blueValue = 0
    totalBlue = 0
    redValue = 0
    totalRed = 0
    yellowValue = 0
    totalYellow = 0
    for key in dicComponent.keys():
        if dicComponent[key][1] == 0.0:
            blueValue = 1
            totalBlue += 1
        elif dicComponent[key][1] == 0.6:
            redValue = 1
            totalRed += 1 
        elif dicComponent[key][1] == 1.0:
            yellowValue = 1
            totalYellow += 1

    #print("total colors: ", blueValue, redValue, yellowValue)
    #print(dic)
    #print(dicComponent, "\n")
    totalColors = blueValue + redValue + yellowValue
    #print("heuristic: ", totalColors -1)
    return int(totalColors - 1)


# Check if a neighbor should be added to open list
def add_to_open(open_list, neighbor):
    for node in open_list:
        if (neighbor == node and neighbor.f >= node.f):
            return False
    return True


def astar1(root, g):
    open_list = [root]
    closed_list = []
    pathDict = {}
    expansion = 0
    while len(open_list) > 0:
        open_list.sort()
        #print("sorted list: ", [n.f for n in open_list])
        node = open_list.pop(0)
        
        
        closed_list.append(node)
        pathDict[node.id] = node.history
        if len(node.dic) <= 1:
            print("length queue: ", len(open_list))
            print("node expansions: ", expansion)
            print("found solution node! ", node.id, node.history)
            break;
            
        print("expanding a noooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooode!!!")
        children = g.expand_node(node)
        #print("queue children: ", children)
        expansion += 1
        selectedChildren = []
        for child in children:
            if child in closed_list:
                continue
                
            child.parent = node
            child.g = len(child.id.split(".")) - 1
            child.h = hColors(child.dic, child.dicComponent)
            child.f = child.g + child.h
            #print("child value: ", child.f, "node value: ", node.f)
            if(add_to_open(open_list, child) == True):
                # Everything is green, add child to open list
                open_list.append(child)
               
    
    
    path = node.id.split(".")
    print(path)
    optimalPath = []
    path_ = copy.deepcopy(path)
    for i in range(len(path_) - 1):
        nodeID = '.'.join(map(str, path)) 
        optimalPath.append(pathDict[nodeID])
        path.pop(-1)
        
    optimalPathSorted = optimalPath[::-1]
    print("optimal: ", optimalPathSorted)
    return [len(optimalPathSorted), expansion]

    
def astar2(root, g):
    open_list = [root]
    closed_list = []
    pathDict = {}
    expansion = 0
    while len(open_list) > 0:
        open_list.sort()
        #print("sorted list: ", [n.f for n in open_list])
        node = open_list.pop(0)
        
        
        closed_list.append(node)
        pathDict[node.id] = node.history
        if len(node.dic) <= 1:
            print("length queue: ", len(open_list))
            print("node expansions: ", expansion)
            print("found solution node! ", node.id, node.history)
            break;
            
        print("expanding a noooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooode!!!")
        children = g.expand_node(node)
        #print("queue children: ", children)
        expansion += 1
        selectedChildren = []
        for child in children:
            if child in closed_list:
                continue
                
            child.parent = node
            child.g = len(child.id.split(".")) - 1
            child.h = hComponents(child.dic, child.dicComponent)
            child.f = child.g + child.h
            #print("child value: ", child.f, "node value: ", node.f)
            if(add_to_open(open_list, child) == True):
                # Everything is green, add child to open list
                open_list.append(child)
               
    
    
    path = node.id.split(".")
    print(path)
    optimalPath = []
    path_ = copy.deepcopy(path)
    for i in range(len(path_) - 1):
        nodeID = '.'.join(map(str, path)) 
        optimalPath.append(pathDict[nodeID])
        path.pop(-1)
        
    optimalPathSorted = optimalPath[::-1]
    print("optimal: ", optimalPathSorted)
    return [len(optimalPathSorted), expansion]

@timeout()
def bfs(root, g):
    queue = [root]
    pathDict = {}
    expansion = 0
    while True:
        
        node = queue.pop(0)
        
        pathDict[node.id] = node.history
        if len(node.dic) <= 1:
            print("length queue: ", len(queue))
            print("node expansions: ", expansion)
            print("found solution node! ", node.id, node.history)
            break;
        g.duplicate_list = []
        children = g.expand_node(node)
        print("expanding a noooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooode!!!")
        expansion += 1
        if expansion == 1:
            print("chiiiiiiiiiiiiiiiiiiiiiildren: ", [[child.id, len(child.dic)] for child in children])
        queue.extend(children)
    
    path = node.id.split(".")
    print(path)
    optimalPath = []
    path_ = copy.deepcopy(path)
    for i in range(len(path_) - 1):
        nodeID = '.'.join(map(str, path)) 
        optimalPath.append(pathDict[nodeID])
        path.pop(-1)
        
    optimalPathSorted = optimalPath[::-1]
    print("optimal: ", optimalPathSorted)
    return [len(optimalPathSorted), expansion]



# A function to perform a Depth-Limited search 
# from given source 'src' 
def DLS(src,maxDepth, counter, g):
    g.duplicate_dictionary = {}
    g.duplicate_list = []
    if len(src.dic) == 1 :
        optimalPath = []
        while src.parent is not None:
            optimalPath.append(src.history)
            src = src.parent
    
        optimalPathSorted = optimalPath[::-1]
        print("optimal: ", optimalPathSorted)
        return [len(optimalPathSorted), counter.iteration]

    # If reached the maximum depth, stop recursing. 
    if maxDepth <= 0 : return 'cutoff'
    
    cutoff_occured = False
    children = g.expand_node(src)
    print("expanding....")
    print("children: ", children)
    counter.iteration += 1
    if counter.iteration == 1:
            print("chiiiiiiiiiiiiiiiiiiiiiildren: ", [[child.id, len(child.dic)] for child in children])
    # Recur for all the vertices adjacent to this vertex 
    for child in children:
        child.parent = src
        result = DLS(child, maxDepth-1, counter, g)
        if(result == 'cutoff'):
            cutoff_occured = True
        elif result is not None:
            return result
    return 'cutoff' if cutoff_occured else 'Not found'
  

def iddfs(src, g):
    counter = Counter()
    maxDepth = len(src.dic)
    if maxDepth == 0 :
        return [0, 0]
    for depth in range(maxDepth):
        g.duplicate_dictionary = {}
        g.duplicate_list = []
        print("Checking with depth: ", depth)
        result = DLS(src, depth, counter, g)
        if result == 'cutoff': 
            print ("Target is NOT reachable from source " +
                "within max depth " + str(depth))
            #maxDepth += 1
        else:
            print ("Target is reachable from source " +
                "within max depth " + str(depth))
            return result
                




def main(algorithm, optimization):
    optimal_moves = [2, 3, 4, 3, 2, 3, 3, 4, 4, 2, 2, 2, 2, 0, 2, 3, 4, 3, 2, 4, 2, 2, 3, 4, 4, 3, 2, 3, 4, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3]
    alg_moves = []
    node_expansions = []

    correct = 0

    timeList = []




    f = open('run_time_%s-%s-PROBLEM50.txt' % (algorithm, optimization), "w")
    f1 = open('node_expansions_%s-%s-PROBLEM50.txt' % (algorithm, optimization), "w")

    correct = 0
    for integer in range(1,2):
        #integer += 1
        problem_file = "problem%s.txt" % (integer)
        #integer -= 1
        g = grid(problem_file, optimization)
        print("\n", problem_file)
        print("==============================")
        print(g.getGrid())
        g.initNeighborGraph()
        print("\n")
        dic, dicComponent = g.getConComponents()
        
        root = Node("0", dic, dicComponent, 0, 0, True)
        
        start = round(time.time()* 1000)
        
        
        
        
        #algorithm
        run = eval(algorithm)
        shortest_length = run(root, g)
        alg_moves.append(shortest_length[0])
        node_expansions.append(shortest_length[1])
        string = str(shortest_length[1]) + "\n"
        f1.write(string)
        
        if optimal_moves[integer] == shortest_length[0]:
            correct += 1
        totalTime = round(time.time()* 1000) - start
        timeList.append(totalTime)
            #f.write('It took {0:0.1f} milliseconds to solve {file} \n'.format(totalTime, file=problem_file))
        string = str(totalTime) + "\n"
        f.write(string)
        

    averageTime = sum(timeList) / len(optimal_moves)
    #f.write('Average Time for solving all {listLength} problem files is: {average}'.format(listLength=len(optimal_moves), average=averageTime))
    f.close()
    f1.close()

    convert_from_txt_to_csv('run_time_%s-%s-PROBLEM50.txt' % (algorithm, optimization), 'run_time')

    convert_from_txt_to_csv('node_expansions_%s-%s-PROBLEM50.txt' % (algorithm, optimization), 'node_expansions')

    winRate = (correct / len(optimal_moves)) * 100


    print("%s Win rate: " % (algorithm), winRate, "%")
    print("%s moves: " % (algorithm), alg_moves)
    print("%s node_expansions: "% (algorithm), node_expansions)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        algorithm = sys.argv[1]
        optimization = sys.argv[2]
        
        main(algorithm, optimization)
        
    else:
        print("Usage: python3.6 grid-kami [algorithm] [optimization]")


