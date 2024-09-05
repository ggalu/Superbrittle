#!/home/gcg/miniforge3/envs/lmpd/bin/python3
import sys
import gmsh
import numpy as np
import copy
from progressbar import progressbar
import math

debug = False
read_from_file = sys.argv[1]

class make_geometry():

    def mesh_geometry(self):
        if read_from_file != "SPALL":
            gmsh.open(read_from_file)
            #gmsh.model.mesh.refine()
            #gmsh.model.mesh.refine()
            #gmsh.model.mesh.refine()
            #gmsh.model.mesh.refine()
        else:
            meshsize = float(sys.argv[2])
            print(f"... creating spall mesh with meshsize f{meshsize}")

            gmsh.model.add("spall")
            width = 0.5
            height = 1.0
            tag = 0
            gmsh.model.occ.add_rectangle(-width/2,-height/2,0,width,height, tag) # id is 1000
            gmsh.model.occ.translate([(2, tag)], 0.5, 0.5, 0) # translate to 0.5,0.5
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [tag], tag, name="My surface 1")


            width = 1.0
            height = 1.0
            tag = 1
            gmsh.model.occ.add_rectangle(-width/2,-height/2,0,width,height, tag) # id is 1000
            gmsh.model.occ.translate([(2, tag)], 1.5, 0.5, 0) # translate to 0.5,0.5
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [tag], tag, name="My surface 2")
        
            N = 8 ## 1.e-2 / N = 8 will give approx. 2.2 million elements
            gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
            gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
            gmsh.model.mesh.generate(self.DIM)    
        gmsh.model.mesh.createEdges()

    def init_data_structures(self):
        # get coordinates
        self.nodes, self.coords, _ = gmsh.model.mesh.getNodes()
        self.coords = np.reshape(self.coords, (-1,3)) # reshape the array to have rows of x,y,z coordinates
        
        if self.DIM ==2: self.coords = np.delete(self.coords, 2, 1) #remove z coordinates if in 2D

        #for i in range(len(self.nodes)):
        #    print("node:", self.nodes[i],  "has coords ", self.coords[i])

        
        eleTypes, self.eleTags, self.eleNodes = gmsh.model.mesh.getElements(self.DIM) # 2 is element type triangle, 3 is tet
        #print(self.eleTags)

        assert len(self.eleTags) == 1 # we only request data for a single element type, eithe ra tri or a tet
        assert len(self.eleNodes[0]) == len(self.eleTags[0]) * (self.DIM + 1) # each tet has 4 nodes, each tri has 3 nodes


        self.numEle = len(self.eleTags[0])
        self.numNodes = self.coords.shape[0]
        self.newNodes = np.zeros((self.numNodes, 2), dtype=np.float32) 
        self.newElems = np.zeros((self.numEle, 3), dtype=np.int32)
        self.nodeVolumes = np.zeros(self.numNodes, dtype=np.float32)
        self.bctype = np.zeros(self.numNodes, np.int32)
        self.bcvel = np.zeros(3, dtype=np.float32)
        self.numVert = 2*self.coords.shape[0]

    def write_data(self):
        import pickle

        # Open a file and use dump() 
        with open('mesh.pkl', 'wb') as file: 
            pickle.dump((self.newNodes, self.newElems, self.elementDataValues, self.nodePhysicalGroup,
                         self.edgeMap), file)
            print("... dumped mesh.pkl ")
    
    def convert_to_array(self, dict, default_value=-1):
        """
        convert a dictionary of lists to array
        """
        longest = max(len(item) for item in dict.values()) # we first find the length of the longest sequence
        num_keys = len(dict.keys())

        arr = -np.ones((num_keys, longest), dtype=np.int32)
        for key in dict:
            sequence = dict[key]
            for i in range(len(sequence)):
                arr[key,i] = sequence[i]

        return arr
    
    def compute_nodal_volumes(self):
        totalMeshVolume = 0.0
        for i in range(self.numEle):
            eTag = self.eleTags[0][i]
            volume = gmsh.model.mesh.getElementQualities([eTag], "volume")[0]
            
            totalMeshVolume += volume
            n1, n2, n3 = int(self.eleNodes[0][i*3]) - 1, int(self.eleNodes[0][i*3+1]) - 1, int(self.eleNodes[0][i*3+2]) - 1
            self.newNodes[n1,:] = self.coords[n1]
            self.newNodes[n2,:] = self.coords[n2]
            self.newNodes[n3,:] = self.coords[n3]
            self.newElems[i,:] = n1, n2, n3

            self.nodeVolumes[n1] += volume/3
            self.nodeVolumes[n2] += volume/3
            self.nodeVolumes[n3] += volume/3
        print(f"total mesh volume is {totalMeshVolume}")

    def order_ascending(self, n1, n2):
        """ return the integers n1, n2 in ascending order
        """
        if n1 < n2:
            return n1, n2
        else:
            return n2, n1
        
    def compute_edge_list(self):
        """
        - compute a list of edges, giving their edge nodes, and their connected elements
        - see https://gmsh.info/doc/texinfo/gmsh.html#x7
        """

        #################################################################################
        #
        # get elements connected to internal edges and save as list edge2elements
        #
        #################################################################################
        elementType = gmsh.model.mesh.getElementType("triangle", 1)
        edgeNodes = gmsh.model.mesh.getElementEdgeNodes(2) # 2 is element type triangle, 3 is tet
        edgeTags, edgeOrientations = gmsh.model.mesh.getEdges(edgeNodes)
        elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)

        # ------------------------------------------------------------------
        # step one: find edges on surface of body, as defined by having only one element connected to the edge, not two
        # ------------------------------------------------------------------
        edges2Elements = {}
        for i in range(len(edgeTags)): # 3 edges per triangle
            edgeTag = edgeTags[i]
            elId = elementTags[i // 3] # 3 edges per triangle
            if not edgeTag in edges2Elements:
                edges2Elements[edgeTag] = [elId] 
            else:
                edges2Elements[edgeTag].append(elId)
        
        isEdgeOnSurface = {}
        for edge in edges2Elements.keys():
            numElemsPerEdge = len(edges2Elements[edge])
            if numElemsPerEdge == 1:
                isEdgeOnSurface[edge] = True
            elif numElemsPerEdge == 2:
                isEdgeOnSurface[edge] = False
            else:
                print(f"Error. numElemsPerEdge should be 1 or 2, but it is {numElemsPerEdge}")

        # ------------------------------------------------------------------
        # step two: create node array with info if node is on surface or not
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # step two: create element dictionary which holds edges and edge surface status
        #           also create edge dictionary which list the node indices for a given edge
        # ------------------------------------------------------------------
        el2Edges = {}
        edge2Nodes = {}
        self.isNodeOnSurface = np.zeros(self.numNodes, dtype=np.int32)

        for i in range(len(edgeTags)): # 3 edges per triangle
            #n1, n2 = order_ascending(int(edgeNodes[2*i]) - 1, int(edgeNodes[2*i+1]) - 1) # subtract 1 to go to zero-based offset
            n1, n2 = int(edgeNodes[2*i]) - 1, int(edgeNodes[2*i+1]) - 1
            elId = elementTags[i // 3] # 3 edges per triangle
            edgeId = edgeTags[i]

            if isEdgeOnSurface[edgeId]:
                #print("edge is in surface!")
                self.isNodeOnSurface[n1] = 1
                self.isNodeOnSurface[n2] = 1

            if not elId in el2Edges:
                el2Edges[elId] = [edgeId]
            else:
                el2Edges[elId].append(edgeId)

            edge2Nodes[edgeId] = (n1, n2) # nodes are zero-offset

        # print accumuldated element information
        #for elId in el2Edges.keys():
        #    print(f"elId: {elId} has edges: {el2Edges[elId]}")
        #    for edge in el2Edges[elId]:
        #        if not isEdgeOnSurface[edge]:
        #            print(f"edge {edge}: nodes {edge2Nodes[edge]}, internalEdge={isEdgeOnSurface[edge]}")
        #sys.exit()

        visited = {}

        numEdges = len(edgeTags)
        edgeMap = [] #-np.ones((numEdges//2, 4), dtype=np.int32)
        for i in range(len(edgeTags)): # this contains duplicate edges, n1--n2 and n2--n1
            n1, n2 = int(edgeNodes[2*i]) - 1, int(edgeNodes[2*i+1]) - 1
            n1, n2 = self.order_ascending(n1, n2)
            key = (n1, n2)
            if key in visited.keys(): # check if we already visited this edge
                pass
            else:
                visited[key] = True
                edgeTag = edgeTags[i]
                connected_elements = edges2Elements[edgeTag] # connected elements to this edge
                el1 = connected_elements[0] - 1 # go to zero-based offset 
                if len(connected_elements) == 2:
                    el2 = connected_elements[1] - 1 # go to zero-based offset 
                else:
                    el2 = -1

                edgeMap.append((n1, n2, el1, el2))
        self.edgeMap = np.asarray(edgeMap, dtype = np.int32)



    def get_Node_Physical_Group(self):


        pg = gmsh.model.getPhysicalGroups(self.DIM) # get physical groups for dimension 2
        print(f"physical node groups of dim={self.DIM}:", pg)

        assign_phyical_groups = False
        if len(pg) > 1: #there is more than 1 physical group present, so assign physical groups to nodes
            assign_phyical_groups = True

        eleTypes_, eleTags, eleNodes_ = gmsh.model.mesh.getElements(self.DIM)

        
        self.nodePhysicalGroup = np.zeros(self.numNodes, dtype=np.int32)
        for i in range(len(eleTags[0])):
            eTag = eleTags[0][i]

            eleType, nodeTags, dim, tag = gmsh.model.mesh.getElement(eTag)
            if assign_phyical_groups:
                physicalGroup = gmsh.model.getPhysicalGroupsForEntity(dim, tag)[0]
                #print(physicalGroup)
            else:
                physicalGroup = 1
            #if physicalGroup != "1": print("physicalGroup", physicalGroup)

            #if debug: print("tri %d: N1=%d, N2=%d, N3=%d" % (eTag, eleNodes[0][i*3], eleNodes[0][i*3+1], eleNodes[0][i*3+2]))
            numNodesThisElement = len(nodeTags)

            ## nodes of this element
            for j in range(numNodesThisElement):
                node = int(nodeTags[j] - 1)
                self.nodePhysicalGroup[node] = physicalGroup

    def getElementPartID(self):
        """
        There are two ways to define the element part ids:
        
        1) If the .msh file was created by the LS-Dyna to gsmh script, a view is present which contains the part ids.

        2) If no views are present, the physical group id is used instead.
        """


        views = gmsh.view.getTags() # shoudl return only one view. if it is not present, create one with the default part ID of 1 for all particles
        print("views: ", gmsh.view.getTags())
        if len(views) == 0:
            print("there are no views, retrieving element part ids from physical groups.")
            pg = gmsh.model.getPhysicalGroups(self.DIM) # get physical groups for dimension 2
            print(f"physical groups of dim={self.DIM}:", pg)
            
            eleTypes_, eleTags, eleNodes_ = gmsh.model.mesh.getElements(self.DIM)
            self.elementDataValues = np.zeros(len(eleTags[0]))
            for i in range(len(eleTags[0])):
                eTag = eleTags[0][i]
                eleType, nodeTags, dim, tag = gmsh.model.mesh.getElement(eTag)
                physicalGroup = gmsh.model.getPhysicalGroupsForEntity(dim, tag)[0]
                self.elementDataValues[int(eTag) - 1] = physicalGroup
            

        else:
            assert len(views) >= 1 # the .msh file created by lsdyna2gmsh.py shoudl have only one field, containing the part id for each element

            selectedView = 1 # for microstructpy, phases are encoded in view 2. For LS-Dyna use 1

            data = gmsh.view.get_model_data(selectedView, 0)
            #print("shape of element data:", data)
            elementTags = data[1]
            elementDataValues = np.asarray(np.asarray(data[2]).flatten())
            #print("shape of element data values:", elementDataValues.shape)
            self.elementDataValues = elementDataValues.astype(np.int32)
            
            if self.elementDataValues.min() != 0:
                print(f"Part IDs do not start with 0 but with {self.elementDataValues.min()}.")
                print("Applying offset such that they start with zero.")
                self.elementDataValues -= self.elementDataValues.min()
            #print("self.elementDataValues", self.elementDataValues)




    def __init__(self, DIM=2):
        self.DIM = DIM
        self.maxNeigh = 0
        self.mapn2m = {}
        self.edgeDict = {}
        self.mapn2el = {}
        gmsh.initialize()
        self.mesh_geometry()
        gmsh.fltk.run()

        self.init_data_structures()
        self.compute_edge_list()
        self.compute_nodal_volumes()
        self.getElementPartID()
        self.get_Node_Physical_Group()
        self.write_data()

        
        gmsh.finalize()

test = make_geometry()