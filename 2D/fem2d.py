
import taichi as ti
import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from matplotlib import cm
from matplotlib.colors import rgb2hex
from enum import IntEnum
import meshio
from scipy import stats



vec = ti.math.vec2
vec2i = ti.types.vector(2, int)
mat2 = ti.math.mat2
retvalType = ti.types.struct(P=mat2, phi=float)

class ElementState(IntEnum):
    FAILED = 0
    INTACT = 1
    FREE = 2

class BoundaryParticleState(IntEnum):
    IS_EDGE = 1
    IS_DUSTBALL = 2

@ti.dataclass
class PartProperty:
    rho: float
    E: float
    nu: float
    mu: float
    la: float
    epsfail: float
    Gf: float
    c0: float
    assigned: int


@ti.dataclass
class BoundaryParticle:
    pos: vec  # Position
    f : vec # force
    vel : vec # velocity
    normal: vec # outwards facing normal
    elid : ti.i32 # node1 of the edge to which this boundary particle belongs
    nn: ti.i32
    n1: ti.i32 # edge is defined between n1 and n2
    n2: ti.i32 # edge is defined between n1 and n2
    n3: ti.i32 # index of the node defining the edge normal
    cornerIndex: ti.i32 # marks where inside the triangle the BP is placed
    state: ti.i32
    r: float

@ti.dataclass
class Vertex:
    pos: vec  # Position
    pos0: vec  # Position in reference
    m: float  # Mass
    v: vec  # Velocity
    f: vec  # Force
    damage: float
    material: ti.i32
    isFree: ti.i32 # 0 if particle is connected to any element
    group: ti.i32

@ti.dataclass
class Elem:
    a: ti.int32 # node index 1 of this triangle
    b: ti.int32 # node index 2 of this triangle
    c: ti.int32 # node index 2 of this triangle
    state: ti.int32 # 1=intact, 0=failed
    part: ti.int32
    maxEig: float
    Gf : float
    lc : float # characteristic length
    damage : float
    edge_ab : ti.int32 # 0 if connected to other elements along this edge, nonzero otherwise
    edge_ac : ti.int32 # 0 if connected to other elements along this edge, nonzero otherwise
    edge_bc : ti.int32 # 0 if connected to other elements along this edge, nonzero otherwise
    dustPos: vec # after failure, this is the position of the element dust cloud
    dustVel: vec # after failure, this is the velocity of the element dust cloud
    dustForce: vec # after failure, this is the force on the element dust cloud
    r : float # element incenter radius
    W : float
    B : mat2 # element shape matrix
    stress: mat2
    F: mat2
    mass: float

@ti.dataclass
class Grid:
    vel: vec
    force: vec
    m: float

@ti.data_oriented
class FEM():
    def __init__(self, materialProperties, initialVelocities, renderParams, contactProperties, runProperties,
                 prescribedVelocities):

        #sys.tracebacklimit=0
        #ti.init(arch=ti.cpu, debug=True, offline_cache=False, default_fp=ti.f32, cpu_max_num_threads=1)
        ti.init(arch=ti.gpu, debug=False, kernel_profiler=False, default_fp=ti.f32)

        self.materialProperties = materialProperties
        self.initialVelocities = initialVelocities
        self.renderParams = renderParams
        self.contactProperties = contactProperties
        self.runProperties = runProperties
        self.prescribedVelocities = prescribedVelocities

        

        self.spall = 1
        self.compact_tension = 0

        self.dimension = 2
        self.gravity = 0.00
        
        self.init_data_fields()
        self.assignInitialVelocitiesToParts()
        self.init_materials()
        self.precomputeTriMat()
        print("maximum element incenter diameter", 2*self.rMax[None])
        print("maximum characteristic element length", self.hMax[None])
        print("minimum characteristic element length", self.hMin[None])
        print("max. C0", self.maxC0)
        self.dtInitial = self.runProperties["dtCFL"] * self.hMin[None] / self.maxC0
        #self.dtInitial = 2.7983469092833673e-08 #!!! OVERRIDE 
        print("stable timestep:", self.dtInitial)
        
        
        self.createFractureSurface()
        self.createPinballs()
        #self.setup_neighbour_list()
        self.setup_linked_list_PBF()

        # set up GUI window
        aspectratio = self.simDomain[0] / self.simDomain[1]
        self.windowX, self.windowY = 1000, 1000 #int(1000 / aspectratio)
        self.gui = ti.GUI(f"number of elements: {self.NE}", (self.windowX, self.windowY))
        self.substeps = 200
        self.time_elapsed = ti.field(float, shape=()); self.time_elapsed[None] = 0.0
        self.dt[None] = 0.0
        self.first_log = True
        self.step = 0
        time_begin_simulation = time.perf_counter()
        
        lastSaveTimeVTU = lastRenderTime = 0.0
        self.nSnapshot = 0
        current_time = 0.0
        while self.gui.running and self.time_elapsed[None] < self.runProperties["runDuration"]:
            self.time_for_one_step = time.perf_counter() # used to measure performance, evaluated in print_line()
            self.step += 1
            for e in self.gui.get_events(self.gui.PRESS):
                if e.key == self.gui.ESCAPE:
                    self.gui.running = False
            
            for i in range(self.substeps):
                currentFailure = self.computeFEM()

                #if self.contactDt[None] < self.dt[None]:
                #    print(f"contact dt is {self.contactDt[None]}, usual dt is {self.dt[None]}")

                if currentFailure: # need to rebuild the list of boundary particles
                    #print("recreating BPs")
                    self.createFractureSurface()
                    self.createPinballs()
                    if self.contactProperties["includeDustBalls"]: self.addFailedElemsToBoundaryParticles()
                else:
                    #print("not updating NL")
                    self.updateBoundaryParticles()

                #self.contact_all_pairs()
                if i % 10 == 0 or currentFailure == 1: self.update_neighbour_list_PBF()
                self.contact_PBF()
                self.map_boundary_force_to_nodes()

                if self.numBoundaryParticles[None] > self.maxNumBoundaryParticles:
                    print("maximum number of boundary particles exceeded!") 
                    sys.exit()
                
                current_time = self.time_integrate()
            
            self.print_status_line()
            if self.runProperties["intervalVTU"] > 0:
                if current_time - lastSaveTimeVTU >= self.runProperties["intervalVTU"]:
                    lastSaveTimeVTU = current_time
                    self.write_VTK()
            
            if self.runProperties["intervalRender"] > 0:
                if current_time - lastRenderTime >= self.runProperties["intervalRender"]:
                    self.render()
                    self.gui.show()
                    lastRenderTime = current_time
                #self.gui.show(f'output/{self.step:06d}.png')


        outmost_loop_time = time.perf_counter() - time_begin_simulation
        

        ti.profiler.print_kernel_profiler_info()

        print("\nOUTMOST LOOP TIME", outmost_loop_time)


    def init_materials(self):
        """
        This function initializes material properties for different parts based on input data and checks
        for any missing assignments.
        """
        #print(self.materialProperties)
        
        self.maxYoungsE = 0.0
        self.maxC0 = 0.0
        for iPart in range(self.PartProps.shape[0]):
            print(f"... assigning material properties for part {iPart}")
            if not iPart in self.materialProperties.keys():
                print(f"ERROR: Part Nr. {iPart} is defined in the mesh, but no material properties exist for this part.")
                sys.exit()

            self.PartProps[iPart].E = self.materialProperties[iPart]["E"]
            self.PartProps[iPart].rho = self.materialProperties[iPart]["rho"]
            self.PartProps[iPart].c0 = ti.sqrt(self.materialProperties[iPart]["E"] / self.materialProperties[iPart]["rho"])
            
            mu = self.materialProperties[iPart]["E"] / (2 * (1 + self.materialProperties[iPart]["nu"]))
            la = self.materialProperties[iPart]["E"] * self.materialProperties[iPart]["nu"] / ((1 + self.materialProperties[iPart]["nu"]) * (1 -2 * self.materialProperties[iPart]["nu"]))
            self.PartProps[iPart].mu = mu
            self.PartProps[iPart].la = la

            self.maxYoungsE = max(self.maxYoungsE, self.materialProperties[iPart]["E"])
            self.maxC0 = max(self.maxC0, self.PartProps[iPart].c0)

            self.PartProps[iPart].epsfail = self.materialProperties[iPart]["epsfail"]

            Gf = 0.5 * self.materialProperties[iPart]["epsfail"]**2 * self.materialProperties[iPart]["E"] * self.averageCharacteristicLength 
            self.PartProps[iPart].Gf = Gf
            self.PartProps[iPart].assigned = 1
            print("Jolly good")

        for i in range(self.PartProps.shape[0]):
            if self.PartProps[i].assigned == 0:
                print(f"ERROR: Part Nr. {i} has not been assigned material properties")
                sys.exit()



    @ti.kernel
    def createFractureSurface(self):
        """
        The function `createFractureSurface` updates edge marks for elements based on their state and
        connectivity, identifying surface edges connected to intact elements.

        This routine updates marks the edges of each element as:
            0 : edge is internal
            1 : edge is on the surface

        It is only neccessary to run this once at the beginning and if an element is failed.
        """

        count = 0
        for iEdge in range(self.edgeMap.shape[0]):
            el1, el2 = self.edgeMap[iEdge,2], self.edgeMap[iEdge,3] # get element ids connected to this edge

            if el2 > -1:
                if self.elems[el1].state != self.elems[el2].state: # one element is intact, the other one failed.
                    # order them such that intact element is at left position
                
                    elintact = el1
                    if self.elems[el2].state > 0: # determine which element ist the intact one
                        elintact = el2

                    self.edgeMap[iEdge,2] = elintact # move intact element to the left
                    self.edgeMap[iEdge,3] = -1 # ... and mark right position as free
                    el2 = -1
                    el1 = elintact


            if el2 == -1 and el1 > -1: # this is a surface edge, as it is only connected to one intact element
                count += 1
                n1, n2 = self.edgeMap[iEdge,0], self.edgeMap[iEdge,1] # nodes of this edge

                ia, ib, ic = self.elems[el1].a, self.elems[el1].b, self.elems[el1].c # nodes of this element

                # need to find that element node which is not in [n1, n2]
                if ia != n1 and ia != n2: 
                    self.elems[el1].edge_bc = 1 # this is the edge not containing ia. mark this edge as active
                elif ib != n1 and ib != n2:
                    nn = ib # --> edge a - c
                    self.elems[el1].edge_ac = 1
                else:
                    nn = ic # --> edge a - b
                    self.elems[el1].edge_ab = 1
        
    @ti.kernel
    def createPinballs(self):
        """
        This subroutine iterates over all intact elements and puts pinballs in those element edges which are on the surface.
        This is only required to if new surface edges have been created since the last time this routine was called.
        """
        self.numBoundaryParticles[None] = 0
        for i in range(self.NE):
            if self.elems[i].state != ElementState.FAILED:
                ia, ib, ic = self.elems[i].a, self.elems[i].b, self.elems[i].c
                if self.elems[i].edge_ab == 1:
                    #self.createPinball(ia, ib, ic, i)
                    self.createPinballMultiple(ia, ib, ic, i)
                if self.elems[i].edge_ac == 1:
                    self.createPinballMultiple(ia, ic, ib, i)
                if self.elems[i].edge_bc == 1:
                    self.createPinballMultiple(ib, ic, ia, i)

                if self.elems[i].edge_ab == self.elems[i].edge_ac == self.elems[i].edge_bc == 1:
                    self.elems[i].state = ElementState.FREE

    def createDustball(self, el1: int):
        """
        This function creates a dustball at a specified element position based on the geometry of the
        surrounding vertices.
        
        @param el1 In the provided code snippet, the `createDustball` method takes an input parameter
        `el1`, which represents the element of `n1`, `n2`, or `nn`. This parameter is used to calculate
        various properties related to the element and create a dustball particle based on those
        """
        # el1 is the element of n1, n2, nn

        n1, n2, n3 = self.elems[el1].a, self.elems[el1].b, self.elems[el1].c

        dab = ti.math.length(self.verts[n2].pos - self.verts[n1].pos)
        dac = ti.math.length(self.verts[n3].pos - self.verts[n1].pos)
        dbc = ti.math.length(self.verts[n3].pos - self.verts[n2].pos)

        incenter = (dbc * self.verts[n1].pos + dac * self.verts[n2].pos + dab * self.verts[n3].pos) / (dab + dac + dbc)
        vel      = (dbc * self.verts[n1].v   + dac * self.verts[n2].v   + dab * self.verts[n3].v  ) / (dab + dac + dbc)

        s = 0.5 * (dab + dac + dbc)
        r = ti.math.sqrt((s - dab) * (s - dac) * (s - dbc) / s)

        i = ti.atomic_add(self.numBoundaryParticles[None], 1) # current index of boundary particle
        self.boundaryParticles[i].pos = incenter
        self.boundaryParticles[i].vel = vel
        self.boundaryParticles[i].r = r
        self.boundaryParticles[i].elid = el1
        self.boundaryParticles[i].state = BoundaryParticleState.IS_DUSTBALL

    @ti.func
    def createPinballMultiple(self, n1: int, n2: int, n3: int, el1: int):
        """
        This function creates boundary particles inside a triangle and at its corners based on given
        input parameters.
        
        @param n1 The `n1` parameter in the `createPinballMultiple` function represents the index of a
        vertex in the `self.verts` array. This vertex is used to calculate the position of a boundary
        particle in the function.
        @param n2 It seems like the information about the `n2` parameter is missing in the provided code
        snippet. If you can provide more context or details about the `n2` parameter, I'd be happy to
        assist you further with understanding or modifying the code.
        @param n3 The `n3` parameter in the `createPinballMultiple` function represents the index of a
        vertex in a triangle. This vertex is used to calculate the positions of the circles that will be
        packed inside the triangle.
        @param el1 It seems like the description of the `el1` parameter is missing. Could you please
        provide more information about what `el1` represents in the context of the
        `createPinballMultiple` function?
        """

        a = self.verts[n1].pos
        b = self.verts[n2].pos
        c = self.verts[n3].pos


        dab = ti.math.length(b - a)
        dac = ti.math.length(c - a)
        dbc = ti.math.length(c - b)

        incenter = (dbc * a + dac * b + dab * c) / (dab + dac + dbc)
        vel      = (dbc * self.verts[n1].v   + dac * self.verts[n2].v   + dab * self.verts[n3].v  ) / (dab + dac + dbc)

        s = 0.5 * (dab + dac + dbc)
        r = ti.math.sqrt((s - dab) * (s - dac) * (s - dbc) / s)
        k = 0.4 # scaling factor for small BP radii

        # 1 / 4: create incenter
        i = ti.atomic_add(self.numBoundaryParticles[None], 1) # current index of boundary particle
        self.boundaryParticles[i].pos = incenter
        self.boundaryParticles[i].vel = vel
        self.boundaryParticles[i].r = 0.999 * r
        self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
        self.boundaryParticles[i].n1 = n1
        self.boundaryParticles[i].n2 = n2
        self.boundaryParticles[i].n3 = n3
        self.boundaryParticles[i].elid = el1
        self.boundaryParticles[i].state = BoundaryParticleState.IS_EDGE
        self.boundaryParticles[i].cornerIndex = 0



        # 2 / 4: create small BP near A
        i = ti.atomic_add(self.numBoundaryParticles[None], 1) # current index of boundary particle
        self.boundaryParticles[i].pos = a + k * (incenter - a)
        self.boundaryParticles[i].vel = vel
        self.boundaryParticles[i].r = k * r
        self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
        self.boundaryParticles[i].n1 = n1
        self.boundaryParticles[i].n2 = n2
        self.boundaryParticles[i].n3 = n3
        self.boundaryParticles[i].elid = el1
        self.boundaryParticles[i].state = BoundaryParticleState.IS_EDGE
        self.boundaryParticles[i].cornerIndex = 1

        # 3 / 4: create small BP near B
        i = ti.atomic_add(self.numBoundaryParticles[None], 1) # current index of boundary particle
        self.boundaryParticles[i].pos = b + k * (incenter - b)
        self.boundaryParticles[i].vel = vel
        self.boundaryParticles[i].r = k * r
        self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
        self.boundaryParticles[i].n1 = n1
        self.boundaryParticles[i].n2 = n2
        self.boundaryParticles[i].n3 = n3
        self.boundaryParticles[i].elid = el1
        self.boundaryParticles[i].state = BoundaryParticleState.IS_EDGE
        self.boundaryParticles[i].cornerIndex = 2

        # 4 / 4: create small BP near C
        i = ti.atomic_add(self.numBoundaryParticles[None], 1) # current index of boundary particle
        self.boundaryParticles[i].pos = c + k * (incenter - c)
        self.boundaryParticles[i].vel = vel
        self.boundaryParticles[i].r = k * r
        self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
        self.boundaryParticles[i].n1 = n1
        self.boundaryParticles[i].n2 = n2
        self.boundaryParticles[i].n3 = n3
        self.boundaryParticles[i].elid = el1
        self.boundaryParticles[i].state = BoundaryParticleState.IS_EDGE
        self.boundaryParticles[i].cornerIndex = 3

    @ti.kernel
    def updateBoundaryParticles(self):
        """
        The function `updateBoundaryParticles` updates the state of boundary particles, including
        dustballs, based on certain conditions and calculations.
        """

        for i in range(self.numBoundaryParticles[None]):
            if self.boundaryParticles[i].state == BoundaryParticleState.IS_EDGE:
                
                n1 = self.boundaryParticles[i].n1
                n2 = self.boundaryParticles[i].n2
                n3 = self.boundaryParticles[i].n3

                # get the associated element nodal positions
                a = self.verts[n1].pos
                b = self.verts[n2].pos
                c = self.verts[n3].pos

                dab = ti.math.length(b - a)
                dac = ti.math.length(c - a)
                dbc = ti.math.length(c - b)

                incenter = (dbc * a + dac * b + dab * c) / (dab + dac + dbc)
                vel      = (dbc * self.verts[n1].v   + dac * self.verts[n2].v   + dab * self.verts[n3].v  ) / (dab + dac + dbc)

                s = 0.5 * (dab + dac + dbc)
                r = ti.math.sqrt((s - dab) * (s - dac) * (s - dbc) / s)
                k = 0.4 # scaling factor for small BP radii

                if self.boundaryParticles[i].cornerIndex == 0: # 1 / 4: create incenter
                    self.boundaryParticles[i].pos = incenter
                    self.boundaryParticles[i].vel = vel
                    self.boundaryParticles[i].r = 0.999 * r
                    self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
                elif self.boundaryParticles[i].cornerIndex == 1: # 2 / 4: create small BP near A
                    self.boundaryParticles[i].pos = a + k * (incenter - a)
                    self.boundaryParticles[i].vel = vel
                    self.boundaryParticles[i].r = k * r
                    self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
                elif self.boundaryParticles[i].cornerIndex == 2: # 3 / 4: create small BP near B
                    self.boundaryParticles[i].pos = b + k * (incenter - b)
                    self.boundaryParticles[i].vel = vel
                    self.boundaryParticles[i].r = k * r
                    self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
                elif self.boundaryParticles[i].cornerIndex == 3: # 4 / 4: create small BP near C
                    self.boundaryParticles[i].pos = c + k * (incenter - c)
                    self.boundaryParticles[i].vel = vel
                    self.boundaryParticles[i].r = k * r
                    self.boundaryParticles[i].normal = self.calculate_normal(n1, n2, n3)
            
            else: # BP is a dustball
                elid = self.boundaryParticles[i].elid
                self.boundaryParticles[i].pos = self.elems[elid].dustPos
                self.boundaryParticles[i].vel = self.elems[elid].dustVel


    @ti.func
    def calculate_normal(self, n1: int, n2: int, n3: int) -> vec:
        """
        This function calculates the normal vector for a given set of vertices in a 2D space.
        
        @param n1 The `n1`, `n2`, and `n3` parameters in the `calculate_normal` function represent
        indices of vertices in a mesh. The function calculates the normal vector orthogonal to the edge
        n1--n2, and pointing away from n3.
        @param n2 .
        @param n3 .
        
        @return A normalized vector representing the normal of the triangle edge defined by the vertices with
        indices n1, n2, and n3.
        """
        v = self.verts[n1].pos - self.verts[n2].pos
        normal = ti.Vector([v[1], -v[0]]) #normals are (-dy, dx) and (dy, -dx).
        normal = normal / (ti.math.length(normal) + 1.0e-6)

        c = self.verts[n3].pos - self.verts[n1].pos
        direction = c.dot(normal)
        if direction > 0:
            normal = -normal

        return normal


    @ti.kernel
    def time_integrate(self) -> float:

        self.prescribedVelocitiesY_force.fill(0.0)
        direction = 1.0

        for i in range(self.numVerts):

            # even after failure of adjacent elements, a vertex needs to have at least 1/3 of its original mass
            # the lacking mass is moved to the dustballs representing the failed elements

            if self.verts[i].m > 0.32 * self.minimumVertexMass[None]:


                acc = self.verts[i].f / self.verts[i].m + ti.Vector([0.0, -self.gravity])
                self.verts[i].v += self.dt[None] * acc

                for k in ti.static(range(len(self.prescribedVelocitiesY_flag))):
                    if self.prescribedVelocitiesY_flag[k] > 0:
                        # check if k-th bit is set, if so, apply prescribed velocity
                        if (self.verts[i].group >> k+1) & 1 == 1: # need to increment k, as first bit is frist bit, not zeroth bit

                            #compute velocity from prescribed acceleration
                            v = self.prescribedVelocitiesY_value[k] * self.time_elapsed[None]
                            #v = self.prescribedVelocitiesY_value[k]
                            #print(f"node {i} belongs to set {k}")
                            self.verts[i].v[1] = v #ramp * self.prescribedVelocitiesY_value[k] * direction
                            ti.atomic_add(self.prescribedVelocitiesY_force[k], self.verts[i].f[1])
                            self.verts[i].f[1] = 0.0
            
                
                self.verts[i].pos += self.dt[None] * self.verts[i].v


            else: # a massless vertex must be a free node and ist mass must be near zero
                #print("mass:", self.verts[i].m, "isFree:", self.verts[i].isFree)
                assert self.verts[i].m < 0.01 *  self.minimumVertexMass[None]
                #assert self.verts[i].isFree == 1
                
            eps = 1.0e-3
            for d in ti.static(range(2)):
                if self.verts[i].pos[d] > self.simDomain[d] - eps:
                    self.verts[i].pos[d] = self.simDomain[d] - eps
                    self.verts[i].v[d] = 0.0
                    #if self.verts[i].v[d] > 0:
                    #    self.verts[i].v[d] *= -1.0
                elif self.verts[i].pos[d] < eps:
                    self.verts[i].pos[d] = eps
                    self.verts[i].v[d] = 0.0
                    #if self.verts[i].v[d] < 0:
                    #    self.verts[i].v[d] *= -1.0

        if self.contactProperties["includeDustBalls"]:
            for i in range(self.NE):
                if self.elems[i].state == ElementState.FAILED:
                    part = self.elems[i].part
                    acc = self.elems[i].dustForce / self.elems[i].mass + ti.Vector([0, -self.gravity])
                    self.elems[i].dustVel += self.dt[None] * acc
                    self.elems[i].dustPos += self.elems[i].dustVel * self.dt[None]

                    eps = 1.0e-3
                    for d in ti.static(range(2)):
                        if self.elems[i].dustPos[d] > self.simDomain[d] - eps:
                            self.elems[i].dustPos[d] = self.simDomain[d] - eps
                            self.elems[i].dustVel[d] *= -1
                        elif self.elems[i].dustPos[d] < eps:
                            self.elems[i].dustPos[d] = eps
                            self.elems[i].dustVel[d] *= -1


        self.time_elapsed[None] += self.dt[None]
        return self.time_elapsed[None]


    def assignInitialVelocitiesToParts(self):
        """ Give all nodes connected to elements with the given material tag the given velocity
        """

        @ti.kernel
        def assignVelocityToElements(part: int, velocity: vec):
            for i in self.elems:
                if self.elems[i].part == part:
                    ia, ib, ic = self.elems[i].a, self.elems[i].b, self.elems[i].c
                    self.verts[ia].v = velocity
                    self.verts[ib].v = velocity
                    self.verts[ic].v = velocity

        for iPart in self.initialVelocities.keys():
            velocity = ti.Vector(self.initialVelocities[iPart])
            assignVelocityToElements(iPart, velocity)    


    


    @ti.kernel
    def copy_vertex_data(self, pos:ti.types.ndarray(), nodesets_bitwise:ti.types.ndarray()):
        
        for i in self.verts:
            for j in ti.static(range(self.dimension)):
                self.verts[i].pos[j] = pos[i,j]
                self.verts[i].pos0[j] = pos[i,j]
            self.verts[i].group = nodesets_bitwise[i]
            assert self.verts[i].group == nodesets_bitwise[i]

    @ti.kernel
    def copy_element_data(self, elems:ti.types.ndarray(), partNums:ti.types.ndarray()) -> float:
        
        avCharacteristicLength = 0.0
        for i in self.elems:
            self.elems[i].a = elems[i,0]
            self.elems[i].b = elems[i,1]
            self.elems[i].c = elems[i,2]
            self.elems[i].state = ElementState.INTACT # 1 = active, 0 = failed
            self.elems[i].part = partNums[i]

            posa = self.verts[self.elems[i].a].pos
            posb = self.verts[self.elems[i].b].pos
            posc = self.verts[self.elems[i].c].pos
            characteristic_length = self.computeTriangleCharacteristicSize(posa, posb, posc)
            avCharacteristicLength += characteristic_length
        avCharacteristicLength /= self.elems.shape[0]
        print("average characteristic element length: ", avCharacteristicLength)
        return avCharacteristicLength

                  

    def init_data_fields(self):

        import pickle
        with open('mesh.pkl', 'rb') as file: 
            pos, np_elems, elemParts, _, edgeMap = pickle.load(file)

        print(f"number of elements: ", len(np_elems))
        print(f"number of nodes: ", len(pos))
        

        nodesets = []; nodesets_bitwise = np.zeros(len(pos), dtype=np.int32)
        if os.path.isfile('nodesets.pkl'):
            with open('nodesets.pkl', 'rb') as file: 
                nodesets, nodesets_bitwise = pickle.load(file)
        print("nodesets", nodesets_bitwise.dtype)
        #sys.exit()

        self.prescribedVelocitiesY_flag = np.asarray(self.prescribedVelocities["Y_flag"], dtype=int)
        self.prescribedVelocitiesY_value = np.asarray(self.prescribedVelocities["Y_value"], dtype=float)
        if len(self.prescribedVelocitiesY_flag) > 0:
            self.prescribedVelocitiesY_force = ti.field(float, shape=self.prescribedVelocitiesY_flag.shape) # this field holds the reaction force due to the prescribed velccity BCs
            assert self.prescribedVelocitiesY_flag.shape == self.prescribedVelocitiesY_value.shape
            print("len nodesets", len(nodesets))
            print("...", self.prescribedVelocitiesY_flag)
            print("len ...", len(self.prescribedVelocitiesY_flag))
            assert len(nodesets) == len(self.prescribedVelocitiesY_flag)
            # ... also need to assert that these nodegroups really exist
        else:
            self.prescribedVelocitiesY_force = ti.field(float, shape=(1,))

        

        self.numParts = len(np.unique(elemParts))
        print(f"Number of parts: {self.numParts}; parts: ", np.unique(elemParts))
        
        if elemParts.min() != 0:
            print(f"Part numbers must start from 0. Currently, the lowest part number is {elemParts.min()}")
            sys.exit()
        if elemParts.max() > self.numParts:
            print("part numbers must be labelled consecutively such that max. part Nr == the number of different parts")
            sys.exit()

        self.PartProps = PartProperty.field(shape=(self.numParts,)) # Parts need to start with zero
        
        # move vertices such that they start at extentPadding
        pos[:,0] -= pos[:,0].min() - self.runProperties["boxPadLeft"]
        pos[:,1] -= pos[:,1].min() - self.runProperties["boxPadBot"]
        simDomain = np.asarray([pos[:,0].max() + self.runProperties["boxPadRight"], pos[:,1].max() + self.runProperties["boxPadTop"]]) # bbox of simulation

        self.simDomain = ti.Vector([simDomain[0], simDomain[1]]) # simulation domain starts at (0,0), so this is the box length

        print("\nSIMULATION DOMAIN:")
        print(f"{0} < {self.simDomain[0]}")
        print(f"{0} < {self.simDomain[1]}")
        print("X extents:", pos[:,0].min(), pos[:,0].max())
        print("Y extents:", pos[:,1].min(), pos[:,1].max())

        
        self.numVerts = len(pos)
        self.NE = len(np_elems)

        self.verts = Vertex.field(shape=(self.numVerts,)) # initialize particles
        self.copy_vertex_data(pos, nodesets_bitwise)

        self.minimumVertexMass = ti.field(float, shape=()); self.minimumVertexMass[None] = 1.0e8
        self.hMin = ti.field(float, shape=()); self.hMin[None] = 1.0e8
        self.hMax = ti.field(float, shape=()); self.hMax[None] = 0.0
        self.dt = ti.field(float, shape=()); self.dt[None] = 0.0
        self.contactDt = ti.field(float, shape=()); self.contactDt[None] = 0.0
        self.rMax = ti.field(float, shape=()); self.rMax[None] = 0.0
        self.numBoundaryParticles = ti.field(ti.i32, shape=())
        self.Epot = ti.field(float, shape=())
        self.contactEnergy = ti.field(float, shape=())
        self.minimumElementMass = ti.field(float, shape=()); self.minimumElementMass[None] = 1.0e22 # minimum mass across all elements
        self.fractureEnergy = ti.field(float, shape=()); self.fractureEnergy[None] = 0.0
        self.failureOccuredNow = ti.field(ti.i32, shape=()); self.failureOccuredNow[None] = 0
        
        self.elems = Elem.field(shape=(self.NE,)) # initialize elements
        self.averageCharacteristicLength = self.copy_element_data(np_elems, elemParts)

        self.edgeMap = ti.field(int, shape=edgeMap.shape)
        self.edgeMap.from_numpy(edgeMap)

        self.maxNumBoundaryParticles = 1024**2
        self.boundaryParticles = BoundaryParticle.field(shape=(self.maxNumBoundaryParticles,))

        


    def render(self):

        gray = int("808080", 0)

        def compute_hex_colors(vizfield):
            minv, maxv = np.min(vizfield), np.max(vizfield)
            cmap_name = "rainbow"
            cmap = plt.get_cmap(cmap_name)
            norm = mpl.colors.Normalize(vmin=minv, vmax=maxv)
            scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
            colors = scalarMap.to_rgba(vizfield)[:,:3]
            hexs = ti.rgb_to_hex(colors.transpose())
            return minv, maxv, hexs

        pos = self.verts.pos.to_numpy()
        pos0 = self.verts.pos0.to_numpy()

        maxDomain = max(self.simDomain[0], self.simDomain[1])
        
        #vizScale = np.asarray((1.0 / self.simDomain[0], 1.0 / self.simDomain[1])) # GUI window only display coordinates in range [0,1], so need to scale physical coordinates
        vizScale = (1.0 / maxDomain, 1.0 / maxDomain)
        radiusScale = vizScale[0] * self.windowX

        
        
        if self.renderParams["renderElements"]: # render elements
            elemState = self.elems.state.to_numpy()
            vizfield = self.elems.maxEig.to_numpy()
            radii = self.elems.r.to_numpy()
            # = maxEig # largest principal stress
            #vizfield = self.elems.material.to_numpy()
            #vizfield = self.elems.lc.to_numpy()
            #vizfield = self.elems.stress.to_numpy()[:,1,1]

            # compute Green-Lagrange strain
            F = self.elems.F.to_numpy()
            vizfield = F[:,0,0]

            minv, maxv, colors = compute_hex_colors(vizfield)

            idx = np.nonzero(elemState == ElementState.INTACT )
            vizfield = vizfield[idx]

            #idx = np.nonzero(vizfield < 1.0e-4)
            minv, maxv, colors = compute_hex_colors(vizfield)
            ai = self.elems.a.to_numpy()[idx] # these are the node indices
            bi = self.elems.b.to_numpy()[idx]
            ci = self.elems.c.to_numpy()[idx]

            a, b, c = pos[ai][:], pos[bi], pos[ci]
            self.gui.triangles(a * vizScale, b * vizScale, c * vizScale, color=colors)
            self.gui.text(f"elems: {minv:.1e} | {maxv:.1e}", pos=(0.1,0.85), font_size=25)

            if True: # output a 1D profile of a quantity along the X-direction
                bc = (a+b+c) / 3.0 # barycenter
                bcx = bc[:,0] # x position of barycenter
                numBins = 300
                bins = np.linspace(0.0, self.simDomain[0], numBins)
                bin_means, bin_edges, binnumber = stats.binned_statistic(bcx, vizfield, statistic='mean', bins=bins)
                filename = "output/profile_%06d" % (self.step)
                np.savetxt(filename, np.column_stack((bin_edges[:-1], bin_means)))


        if self.renderParams["renderElementOutlines"]: # render triangle outlines
            elemState = self.elems.state.to_numpy()
            #elems = self.elems.to_numpy()
            idx = np.nonzero(elemState > 0)
            ai = self.elems.a.to_numpy()[idx]
            bi = self.elems.b.to_numpy()[idx]
            ci = self.elems.c.to_numpy()[idx]
            a, b, c = pos[ai][:] * vizScale, pos[bi] * vizScale, pos[ci] * vizScale

            self.gui.lines(a, b)
            self.gui.lines(a, c)
            self.gui.lines(b, c)

        if self.renderParams["renderFreeVertices"]: # render free vertices
            radii = self.verts.r.to_numpy()
            vertState = self.verts.isFree.to_numpy()
            idx = np.nonzero(vertState > 0)[0]
            if len(idx) > 0:
                self.gui.circles(pos[idx]*vizScale, radius=10*radii[idx]*radiusScale)

        if self.renderParams["renderDustballs"]: # render failed elements using dustballs
            pos = self.boundaryParticles.pos.to_numpy()[:self.numBoundaryParticles[None]]
            radii = self.boundaryParticles.r.to_numpy()[:self.numBoundaryParticles[None]]
            BPState = self.boundaryParticles.state.to_numpy()[:self.numBoundaryParticles[None]]
            idx = np.nonzero(BPState == BoundaryParticleState.IS_DUSTBALL)[0]
            if len(idx) > 0:
                self.gui.circles(pos[idx]*vizScale, radius=radii[idx]*radiusScale, color=gray)

        if self.renderParams["renderVertices"]: # render vertices
            radii = self.verts.r.to_numpy()
            #vizfield = self.verts.dv.to_numpy()[:,1]
            vizfield = pos[:,1] - pos0[:,1]
            idx = np.indices(vizfield.shape)[0]
            if len(idx) > 0:
                minv, maxv, colors = compute_hex_colors(vizfield)
                self.gui.text(f"verts: {minv:.1e} | {maxv:.1e}", pos=(0.1,0.95), font_size=25)        
                self.gui.circles(pos[idx]*vizScale, radius=0.3 * radii[idx]*radiusScale, color=colors)

        if self.renderParams["renderNodeSets"]: # render vertices on which a nodeset is defined
            pos = self.verts.pos.to_numpy()
            group = self.verts.group.to_numpy()
            idx = np.nonzero(group)[0]
            if len(idx) > 0:
                self.gui.circles(pos[idx]*vizScale, radius=2)

        
        if self.renderParams["renderBoundaryParticles"]:# render boundary particles
            print("\nnumber of Pinballs:", self.numBoundaryParticles[None])
            pos = self.boundaryParticles.pos.to_numpy()[:self.numBoundaryParticles[None]]
            normals = self.boundaryParticles.normal.to_numpy()[:self.numBoundaryParticles[None]]
            radii = self.boundaryParticles.r.to_numpy()[:self.numBoundaryParticles[None]]
            vizfield = self.boundaryParticles.nn.to_numpy()[:self.numBoundaryParticles[None]]
            #vizfield = self.boundaryParticles.f.to_numpy()[:self.numBoundaryParticles[None]]
            #vizfield = vizfield[:,0]
            minv, maxv, colors = compute_hex_colors(vizfield)

            self.gui.circles(pos * vizScale, radii*radiusScale, color=colors)
            #self.gui.arrows(orig=pos*vizScale, direction=normals * 0.02)
            self.gui.text(f"BPs: {minv:.1e} | {maxv:.1e}", pos=(0.1,0.95), font_size=25)        
            self.gui.text(f"num BPs: {self.numBoundaryParticles[None]}", pos=(0.1,0.85), font_size=25)
        
        self.gui.text(f"dt: {self.dt[None]:.1e} | time: {self.time_elapsed[None]:.1e}", pos=(0.5,0.98), font_size=25)
    

    @ti.kernel
    def precomputeTriMat(self):
        totalArea = 0.0
        minCharacteristicLength = 1.0e8
        for i in range(self.NE):
            part = self.elems[i].part
            self.elems[i].state = 1
            ia, ib, ic = self.elems[i].a, self.elems[i].b, self.elems[i].c
            a, b, c = self.verts[ia].pos, self.verts[ib].pos, self.verts[ic].pos
            characteristic_length = self.computeTriangleCharacteristicSize(a, b, c)
            ti.atomic_min(minCharacteristicLength, characteristic_length)
            ti.atomic_max(self.hMax[None], characteristic_length)

            M = ti.Matrix.cols([a - c, b - c])
            self.elems[i].B = M.inverse()

            area = 0.5 * ti.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            self.elems[i].W = area
            totalArea += area
            ti.atomic_min(self.minimumElementMass[None], area * self.PartProps[part].rho)
            
            elmass_third = self.PartProps[part].rho * area / 3.0
            ti.atomic_add(self.verts[ia].m, elmass_third)
            ti.atomic_add(self.verts[ib].m, elmass_third)
            ti.atomic_add(self.verts[ic].m, elmass_third)

            self.elems[i].mass = 3.0 * elmass_third

            # determine edge lengths
            dab = ti.math.length(a-b)
            dac = ti.math.length(a-c)
            dbc = ti.math.length(a-c)

            # compute largest incenter radius -- required to set up contact neighbour list
            s = 0.5 * (dab + dac + dbc)
            r = ti.math.sqrt((s - dab) * (s - dac) * (s - dbc) / s)
            ti.atomic_max(self.rMax[None], r)

            
        self.hMin[None] = minCharacteristicLength


        # determine stable contact timestep
        self.contactDt[None] = 2.0 / ti.sqrt(self.contactProperties["contactStiffness"] * self.maxYoungsE / self.minimumElementMass[None])
        print("contact dt:", self.contactDt[None])

        totalMass = 0.0
        self.minimumVertexMass[None] = 1.0e8
        for i in range(self.numVerts):
            totalMass += self.verts[i].m
            ti.atomic_min(self.minimumVertexMass[None], self.verts[i].m)

        print(f"total area of mesh is {totalArea}, total mass of mesh is {totalMass}")


    @ti.func
    def get_piola_corot(self, u, l, F):
        """
        calculate the first Piola-Kirchhoff stress tensor for co-rotational model
        return: Piola stress tensor
        """
        I = ti.Matrix.identity(float, 2)

        # co-rotated 
        U, sig, V = ti.svd(F)
        R = U @ V.transpose()
        P = 2 * u * (F - R) + l * ((R.transpose() @ F).trace() - 2) * R #3d: 2->3
        sigI = sig - I
        sigInorm2 = sigI[0,0]**2 + sigI[0,1]**2 + sigI[1,0]**2 + sigI[1,1]**2 
        sigITr = sigI[0,0] + sigI[1,1]
        phi = u * sigInorm2 + 0.5 * l * sigITr**2

        # small strain linear elasticity
        #eps = 0.5 * (F + F.transpose()) - I
        #P = 2 * u * eps +  l * eps.trace() * I
        #phi = u * (eps[0,0]**2 + eps[0,1]**2 + eps[1,0]**2 + eps[1,1]**2) + 0.5 * l * (eps.trace())**2
        
        retval = retvalType(P=P, phi=phi)
        return retval


    @ti.func
    def computeTriangleCharacteristicSize(self, a, b, c):
        l1 = ti.math.length(a - b) # compute stable time step
        l2 = ti.math.length(a - c)
        l3 = ti.math.length(b - c)
        lMax = ti.max(l1, l2, l3)
        area = ti.abs((a - c).cross(b - c)) # volume (area in 2d
        characteristic_length = 2 * area / lMax
        return characteristic_length
    



    @ti.func
    def eig_sym_2x2(self, a, b, d):
        det = ti.sqrt((a - d)**2 + 4 * b *b)
        ev1 = 0.5 * (a + d + det)
        ev2 = 0.5 * (a + d - det)

        return [ev1, ev2]
    
    # Singular Value Decomposition so that U and V are rotation matrices
    # only applies to 3D
    @ti.func
    def ssvd(self, F):
        U, sig, V = ti.svd(F)
        if U.determinant() < 0:
            for i in ti.static(range(3)): U[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        if V.determinant() < 0:
            for i in ti.static(range(3)): V[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        return U, sig, V

    @ti.func
    def viscous_pairwise_force(self, va, vb, a, b):
        ab = (b - a)
        ab /= ti.sqrt(ab[0]**2 + ab[1]**2) # unit vector between a and b
        vab = (vb - va) @ ab * ab
        fv = self.runProperties["damping"] * vab

        return fv

    @ti.kernel
    def computeFEM(self) -> int:
        
        I = ti.Matrix.identity(float, 2)
        self.failureOccuredNow[None] = 0

        minCharacteristicLength = 1.0e8
        self.Epot[None] = 0.0
        for i in self.verts:
            self.verts[i].f.fill(0.0)
            self.verts[i].isFree = 1

        for i in range(self.NE):
            if self.elems[i].state > 0:
                part = self.elems[i].part

                # deformation gradient
                ia, ib, ic = self.elems[i].a, self.elems[i].b, self.elems[i].c
                self.verts[ia].isFree = 0
                self.verts[ib].isFree = 0
                self.verts[ic].isFree = 0
                a, b, c = self.verts[ia].pos, self.verts[ib].pos, self.verts[ic].pos
                va, vb, vc = self.verts[ia].v, self.verts[ib].v, self.verts[ic].v
                characteristic_length = self.computeTriangleCharacteristicSize(a, b, c)
                self.elems[i].lc = characteristic_length
                dt = self.runProperties["dtCFL"] * characteristic_length / self.maxC0            
                ti.atomic_min(minCharacteristicLength, characteristic_length)

                D_i = ti.Matrix.cols([a - c, b - c])
                #F = D_i @ self.B[i] # deformation gradient
                F = D_i @ self.elems[i].B # deformation gradient
                self.elems[i].F = F
                D_i = ti.Matrix.cols([va - vc, vb - vc])
                Fdot = D_i @ self.elems[i].B # time derivative of deformation gradient
                #L = Fdot @ ti.math.inverse(F) # velocity gradient
                Edot = 0.5 * (Fdot + Fdot.transpose()) # strain rate tensor
                Vdot = 0.5 * Edot.trace() * I # volumetric strain rate tensor
                q1 = self.runProperties["damping"] * self.PartProps[part].rho * self.PartProps[part].c0 * characteristic_length * Edot #Vdot

                #print("F", F)

                C = F.transpose() @ F # left Cauchy-Green tensor
                E = 0.5 * (C - I)
                ev1, ev2 = self.eig_sym_2x2(E[0,0], E[0,1], E[1,1])
                maxEv = ti.max(ev1, ev2)
                
                critStrainLocal = ti.math.sqrt(2 * self.PartProps[part].Gf / (self.PartProps[part].E * characteristic_length))
                self.elems[i].maxEig = maxEv / critStrainLocal
                if maxEv > critStrainLocal: # or dt < 0.01 * self.dtInitial:
                    self.elems[i].damage += 0.1

                D = 1.0 - self.elems[i].damage

                stress = self.get_piola_corot(self.PartProps[part].mu, self.PartProps[part].la, F)
                self.elems[i].stress = stress.P # save PK1 
                ti.atomic_add(self.Epot[None], D * stress.phi * self.elems[i].W)

                #
                # compute nodal elastic forces
                #
                H = -self.elems[i].W * D * (stress.P + q1) @ self.elems[i].B.transpose()
                #print("H", H)

                f1 = ti.Vector([H[0, 0], H[1, 0]])
                f2 = ti.Vector([H[0, 1], H[1, 1]])
                f3 = -f1 - f2

                self.verts[ia].f += f1
                self.verts[ib].f += f2
                self.verts[ic].f += f3

                if D <= 0:

                    self.failureOccuredNow[None] = 1
                    self.elems[i].state = ElementState.FAILED # fail this element
                    ti.atomic_add(self.fractureEnergy[None], stress.phi * self.elems[i].W)

                    # need to remove the mass of this element from its vertices
                    elmass_third = self.elems[i].W * self.PartProps[part].rho / 3.0
                    ti.atomic_sub(self.verts[ia].m, elmass_third)
                    ti.atomic_sub(self.verts[ib].m, elmass_third)
                    ti.atomic_sub(self.verts[ic].m, elmass_third)

                    # initialize the position of this failed element
                    dab = ti.math.length(b - a)
                    dac = ti.math.length(c - a)
                    dbc = ti.math.length(c - b)

                    # incenter radius
                    s = 0.5 * (dab + dac + dbc)
                    r = ti.math.sqrt((s - dab) * (s - dac) * (s - dbc) / s)

                    incenter = (dbc * a  + dac * b  + dab * c)  / (dab + dac + dbc)
                    vel      = (dbc * va + dac * vb + dab * vc) / (dab + dac + dbc)
                    
                    self.elems[i].dustPos = incenter
                    self.elems[i].dustVel = vel
                    self.elems[i].r = r
                    


        #print("minimum h:", minCharacteristicLength)
        #minCharacteristicLength = ti.min(minCharacteristicLength, 0.02)
        #self.dt[None] = self.runProperties["dtCFL"] * minCharacteristicLength / self.maxC0
        self.dt[None] = self.dtInitial

        return self.failureOccuredNow[None]
        


    @ti.kernel
    def addFailedElemsToBoundaryParticles(self):
        for i in range(self.NE):
            if self.elems[i].state == ElementState.FAILED:
                j = ti.atomic_add(self.numBoundaryParticles[None], 1) # current index of boundary particle
                self.boundaryParticles[j].elid = i
                self.boundaryParticles[j].pos = self.elems[i].dustPos
                self.boundaryParticles[j].vel = self.elems[i].dustVel
                self.boundaryParticles[j].state = BoundaryParticleState.IS_DUSTBALL
                self.boundaryParticles[j].r = self.elems[i].r
                

    @ti.kernel
    def map_boundary_force_to_nodes(self):
        """
        Final step of the pinball algorithm.
        - for intact lements, distribute pinball force on all nodes of element
        - for failed elements (dustballs), keep the force on element's dustball
        """
        for i in range(self.numBoundaryParticles[None]):

            elid = self.boundaryParticles[i].elid
            if self.boundaryParticles[i].state == BoundaryParticleState.IS_DUSTBALL:
                self.elems[elid].dustForce = self.boundaryParticles[i].f
            
            else:
                force = self.boundaryParticles[i].f / 3.0

                # distribute the force of this pinball to all nodes of this element
                ia, ib, ic = self.elems[elid].a, self.elems[elid].b, self.elems[elid].c
                ti.atomic_add(self.verts[ia].f, force)
                ti.atomic_add(self.verts[ib].f, force)
                ti.atomic_add(self.verts[ic].f, force)


    @ti.func
    def dustballForce(self, i: int, j: int):

        if self.boundaryParticles[i].state != self.boundaryParticles[j].state:
            # One BP is an edge, the other one a dustball. Find out which is which.
            edge = i
            dustball = j
            if self.boundaryParticles[i].state == BoundaryParticleState.IS_DUSTBALL:
                dustball = i
                edge = j
            
            dx = self.boundaryParticles[dustball].pos - self.boundaryParticles[edge].pos
            radsum = self.boundaryParticles[edge].r + self.boundaryParticles[dustball].r
            r = ti.math.length(dx) + 1.0e-4
            if r < radsum:
                dx *= 1. / r

                n = self.boundaryParticles[edge].normal
                angle = n.dot(dx)
                anglecutoff = 0.0 # 0.0 means 90 degree, only consider interactions above this angle
                angle -= anglecutoff

                if (angle > 0.0):
                    overlap = radsum - r
                    strain = overlap / radsum # = (radsum - r) / radsum = 1 - r / radsum

                    elid = self.boundaryParticles[edge].elid # look up element
                    part = self.elems[elid].part
                    
                    stress = self.PartProps[part].E * strain
                    area = radsum # ad-hoc choice
                    forceMagnitude = self.contactProperties["contactStiffness"] * stress * area 

                    vrel = (self.boundaryParticles[dustball].vel - self.boundaryParticles[edge].vel).dot(n)
                    if vrel > 0: #moving apart
                        forceMagnitude = forceMagnitude * self.contactProperties["restitutionCoefficient"]
                    
                    force = -angle * forceMagnitude * n
                    ti.atomic_add(self.boundaryParticles[edge].f, force)
                    ti.atomic_add(self.boundaryParticles[dustball].f, -force)
            
        else:
            #pass
            # both BPs are dustballs, use isotropic potential
            dx = self.boundaryParticles[j].pos - self.boundaryParticles[i].pos
            radsum = self.boundaryParticles[i].r + self.boundaryParticles[j].r
            r = ti.math.length(dx) + 1.0e-4
            if r < radsum:
                dx *= 1. / r

                #vrel = (self.boundaryParticles[j].vel - self.boundaryParticles[i].vel).dot(dx)
                #rho = 2.0e-6
                #V = radsum * radsum * 1
                #m = rho * V
                #force = (-0.1*self.contactProperties["contactDamping"] * m * vrel * self.maxC0 / radsum) * dx

                overlap = radsum - r
                strain = overlap / radsum


                elid = self.boundaryParticles[i].elid # look up element
                part = self.elems[elid].part

                stress = self.contactProperties["contactStiffness"] * self.PartProps[part].E * strain
                area = radsum
                forceMagnitude = self.contactProperties["contactStiffness"] * stress * area 

                #vrel = (self.boundaryParticles[i].vel - self.boundaryParticles[j].vel).dot(dx)
                #if vrel > 0: #moving apart
                #    forceMagnitude *= self.contactProperties["restitutionCoefficient"]
                
                force = forceMagnitude * dx
                ti.atomic_add(self.boundaryParticles[i].f, -force)
                ti.atomic_add(self.boundaryParticles[j].f, force)

    @ti.func
    def pinballForceIstotropic(self, i: int, j: int):
        # force between two pinballs without edges
        dx = self.boundaryParticles[i].pos - self.boundaryParticles[j].pos
        radsum = self.boundaryParticles[i].r + self.boundaryParticles[j].r
        
        if dx[0]**2 + dx[1]**2 < radsum * radsum:
                
            ni = self.boundaryParticles[i].normal
            nj = self.boundaryParticles[j].normal
            angle = - ni.dot(nj)
            anglecutoff = -0.1 # 0.0 means 90 degree, only consider interactions above this angle
            angle -= anglecutoff

            nij_norm = ti.math.length(nj - ni)
            if nij_norm > 1.0e-4 and angle > 0:
                n = (nj - ni) / nij_norm
                r = ti.math.length(dx)
                overlap = radsum - r
                strain = overlap / radsum

                elidi = self.boundaryParticles[i].elid # look up elements and compute mean stiffness
                elidj = self.boundaryParticles[j].elid # 
                parti = self.elems[elidi].part
                partj = self.elems[elidj].part
                Eeff = 0.5 * (self.PartProps[parti].E + self.PartProps[partj].E)

                stress = self.contactProperties["contactStiffness"] * Eeff * strain * (1 + strain)
                area = radsum
                forceMagnitude = stress * area

                #vrel = (self.boundaryParticles[j].vel - self.boundaryParticles[i].vel).dot(dx)
                #if vrel < 0: # reduce contact force for particles moving away from each other
                #    forceMagnitude = self.contactProperties["restitutionCoefficient"] * forceMagnitude

                force = forceMagnitude * n

                ti.atomic_add(self.boundaryParticles[i].f, force)
                ti.atomic_add(self.boundaryParticles[j].f, -force)


    @ti.func
    def pinballForce(self, i: int, j: int):
        # force between two pinballs with edges
        dx = self.boundaryParticles[i].pos - self.boundaryParticles[j].pos
        radsum = self.boundaryParticles[i].r + self.boundaryParticles[j].r
        
        if dx[0]**2 + dx[1]**2 < radsum * radsum:
            # viscosity makes no difference
            
            ni = self.boundaryParticles[i].normal
            nj = self.boundaryParticles[j].normal
            angle = - ni.dot(nj)
            anglecutoff = -0.1 # 0.0 means 90 degree, only consider interactions above this angle
            angle -= anglecutoff

            nij_norm = ti.math.length(nj - ni)
            if nij_norm > 1.0e-4 and angle > 0:
                n = (nj - ni) / nij_norm
                r = ti.math.length(dx)
                overlap = radsum - r
                strain = overlap / radsum

                elidi = self.boundaryParticles[i].elid # look up elements and compute mean stiffness
                elidj = self.boundaryParticles[j].elid # 
                parti = self.elems[elidi].part
                partj = self.elems[elidj].part
                Eeff = 0.5 * (self.PartProps[parti].E + self.PartProps[partj].E)

                stress = self.contactProperties["contactStiffness"] * Eeff * strain * (1 + strain)
                area = radsum
                forceMagnitude = angle * stress * area

                vrel = (self.boundaryParticles[j].vel - self.boundaryParticles[i].vel).dot(n)
                if vrel < 0: # reduce contact force for particles moving away from each other
                    forceMagnitude = self.contactProperties["restitutionCoefficient"] * forceMagnitude

                force = forceMagnitude * n

                ti.atomic_add(self.boundaryParticles[i].f, force)
                ti.atomic_add(self.boundaryParticles[j].f, -force)


    @ti.kernel
    def contact_all_pairs(self):
        for i in range(self.numBoundaryParticles[None]):
            self.boundaryParticles[i].f.fill(0.0)
        
        for i in range(self.numBoundaryParticles[None] - 1):
            for j in range(i + 1, self.numBoundaryParticles[None]):
                if self.boundaryParticles[i].state == self.boundaryParticles[j].state == BoundaryParticleState.IS_EDGE:
                    self.pinballForce(i, j)
                else:
                    self.dustballForce(i, j)


    def setup_linked_list_PBF(self):

        self.grid_size = 6*self.rMax[None]
        self.grid_n = ti.Vector([int(self.simDomain[0]/self.grid_size), int(self.simDomain[1]/self.grid_size), 1])
        print("Neighbour grid size:", self.grid_n)
        self.numCells = self.grid_n[0] * self.grid_n[1] * self.grid_n[2]
        print("number of neighbor cells:", self.numCells)

        self.list_head = ti.field(dtype=ti.i32, shape=(self.numCells,))
        self.list_cur = ti.field(dtype=ti.i32, shape=(self.numCells,))
        self.list_tail = ti.field(dtype=ti.i32, shape=(self.numCells,))
        self.grain_count = ti.field(dtype=ti.i32, shape=(self.grid_n[0], self.grid_n[1], self.grid_n[2]))
        self.column_sum = ti.field(dtype=ti.i32, shape=(self.grid_n[0], self.grid_n[1]))
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n[0], self.grid_n[1]))
        self.particle_id = ti.field(dtype=ti.i32, shape=(self.maxNumBoundaryParticles,))

        # set up the dynamic neighbour list
        #self.Snode = ti.root.dynamic(ti.i, 1024**2, chunk_size=32)
        # estimate the number of neighbors
        nn = 100
        #self.dynamicNL = ti.field(dtype = vec2i, shape=(self.maxNumBoundaryParticles * nn,))
        self.neighbourListPBF = ti.field(dtype=ti.i32, shape=(self.maxNumBoundaryParticles * nn,2))
        self.numNeighPair = ti.field(dtype=ti.i32, shape=()); self.numNeighPair[None] = 0
        #self.Snode.place(self.dynamicNL)

    @ti.func
    def findNeighbors_PBF(self):
        scale = ti.Vector([self.grid_n[0] / self.simDomain[0], self.grid_n[1] / self.simDomain[1], 1.0])
        
        self.grain_count.fill(0)
        for i in range(self.numBoundaryParticles[None]):
            self.boundaryParticles[i].nn = 0
            pos = ti.Vector([self.boundaryParticles[i].pos[0], self.boundaryParticles[i].pos[1], 0.0])
            pos = pos * scale
            grid_idx = ti.floor(pos, int)

            for d in ti.static(range(3)):
                if not 0 <= grid_idx[d] < self.grid_n[d]:
                    print(self.boundaryParticles[i].pos, pos, grid_idx, self.grid_n[0], self.grid_n[1])
                    assert 0 <= grid_idx[d] < self.grid_n[d]

            self.grain_count[grid_idx] += 1

        self.column_sum.fill(0)
        for i, j, k in ti.ndrange(self.grid_n[0], self.grid_n[1], self.grid_n[2]):        
            ti.atomic_add(self.column_sum[i, j], self.grain_count[i, j, k])
        
        # this is because memory mapping can be out of order
        _prefix_sum_cur = 0    
        for i, j in ti.ndrange(self.grid_n[0], self.grid_n[1]): # skip Z direction
            self.prefix_sum[i, j] = ti.atomic_add(_prefix_sum_cur, self.column_sum[i, j])
        
        for i, j, k in ti.ndrange(self.grid_n[0], self.grid_n[1], self.grid_n[2]):        
            # we cannot visit prefix_sum[i,j] in this loop
            pre = ti.atomic_add(self.prefix_sum[i,j], self.grain_count[i, j, k])        

            #linear_idx = i * grid_n * grid_n + j * grid_n + k
            linear_idx = i + j * self.grid_n[0] + k * self.grid_n[1] * self.grid_n[0]

            self.list_head[linear_idx] = pre
            self.list_cur[linear_idx] = self.list_head[linear_idx]
            # only pre pointer is useable
            self.list_tail[linear_idx] = pre + self.grain_count[i, j, k]       
        
        for i in range(self.numBoundaryParticles[None]):
            pos = ti.Vector([self.boundaryParticles[i].pos[0], self.boundaryParticles[i].pos[1], 0.0])
            pos = pos * scale
            grid_idx = ti.floor(pos, int)

            linear_idx = grid_idx[0] + grid_idx[1] * self.grid_n[0] + grid_idx[2] * self.grid_n[1] * self.grid_n[0]
            grain_location = ti.atomic_add(self.list_cur[linear_idx], 1)
            self.particle_id[grain_location] = i


    @ti.func
    def getNeighborGrid_PBF(self, i):
        scale = ti.Vector([self.grid_n[0] / (self.simDomain[0]), self.grid_n[1] / (self.simDomain[1]), 1.0])
        pos = ti.Vector([self.boundaryParticles[i].pos[0], self.boundaryParticles[i].pos[1], 0.0])
        pos = pos * scale
        grid_idx = ti.floor(pos, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, self.grid_n[0])
        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, self.grid_n[1]) 
        z_begin = max(grid_idx[2] - 1, 0)
        z_end = min(grid_idx[2] + 2, self.grid_n[2])
        return x_begin, x_end, y_begin, y_end, z_begin, z_end 

    @ti.kernel
    def update_neighbour_list_PBF(self):
        

        self.findNeighbors_PBF()

        #self.dynamicNL.deactivate()
        self.numNeighPair[None] = 0
        for i in range(self.numBoundaryParticles[None]):
            x_begin, x_end, y_begin, y_end, z_begin, z_end = self.getNeighborGrid_PBF(i)
            for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin,x_end),(y_begin,y_end),(z_begin,z_end)):            
                neigh_linear_idx = neigh_i + neigh_j * self.grid_n[0] + neigh_k * self.grid_n[1] * self.grid_n[0]
                for p_idx in range(self.list_head[neigh_linear_idx], self.list_tail[neigh_linear_idx]):
                    j = self.particle_id[p_idx]
                    if i > j:
                        if self.boundaryParticles[i].elid != self.boundaryParticles[j].elid:

                            dx = self.boundaryParticles[i].pos - self.boundaryParticles[j].pos
                            cutoff = 2.0 * (self.boundaryParticles[i].r + self.boundaryParticles[j].r)

                            if dx[0]**2 + dx[1]**2 < cutoff**2:
                                ipair = ti.atomic_add(self.numNeighPair[None], 1)
                                self.neighbourListPBF[ipair,0] = i
                                self.neighbourListPBF[ipair,1] = j

                            
    @ti.kernel
    def contact_PBF(self):
        for i in range(self.numBoundaryParticles[None]):
            self.boundaryParticles[i].f.fill(0.0)
        
        for ipair in range(self.numNeighPair[None]):
            i, j = self.neighbourListPBF[ipair,0], self.neighbourListPBF[ipair,1]

            if self.boundaryParticles[i].state == self.boundaryParticles[j].state == BoundaryParticleState.IS_EDGE:
                self.pinballForce(i, j)
            else:
                self.dustballForce(i, j)
            ti.atomic_add(self.boundaryParticles[j].nn, 1)
            ti.atomic_add(self.boundaryParticles[i].nn, 1)

    def print_status_line(self):

        # time taken for one big step:
        self.time_for_one_step = time.perf_counter() - self.time_for_one_step

        # compute linear momentum of vertices
        vel = self.verts.v.to_numpy()
        m = self.verts.m.to_numpy()
        momentum = vel * m[..., np.newaxis]
        linMomentum = np.sum(momentum, axis=0)
        Ekin = 0.5 * np.sum(m * np.einsum('ij,ij->i',vel,vel))

        # compute linear momentum of failed elements which are represented as dustballs
        elState = self.elems.state.to_numpy()
        idx = np.nonzero(elState == ElementState.FAILED)[0]
        if len(idx) > 0:
            partids = self.elems.part.to_numpy()[idx]
            m = self.elems.mass.to_numpy()[idx]
            vel = self.elems.dustVel.to_numpy()[idx]
            momentum = vel * m[..., np.newaxis]

            linMomentum += np.sum(momentum, axis=0)
            Ekin += 0.5 * np.sum(m * np.einsum('ij,ij->i',vel,vel))
        
        # reaction forces due to prescribed velocity BCs
        reactionForcesY = self.prescribedVelocitiesY_force.to_numpy()

        header = ""; line = "\r"
        header += "%16s " % ("elapsed_time")
        line += "%16e " % self.time_elapsed[None]

        header += "%16s " % ("dt")
        line += "%16e " % self.dt[None]

        header += "%16s " % ("Sum_Mx")
        line += "%16e " % linMomentum[0]

        header += "%16s " % ("kinetic_energy")
        line += "%16e " % Ekin

        header += "%16s " % ("potential_energy")
        line += "%16e " % self.Epot[None]

        header += "%16s " % ("contact_energy")
        line += "%16e " % self.contactEnergy[None]

        header += "%16s " % ("fractured_energy")
        line += "%16e " % self.fractureEnergy[None]

        header += "%16s " % ("update time/element")
        line += "%16e " % (self.time_for_one_step / (self.substeps * self.NE))

        if self.first_log == True:
            self.first_log = False
            with open("log.dat", "w") as f:
                f.write("#" + header)
            print("\n", header)

            print("reactionForcesY", reactionForcesY)
            with open("reaction_forces.dat", "w") as f:
                header = "%16s " % "time"
                for i in range(len(reactionForcesY)):
                    header += "%16s " % f"nodeset_{i}_RFY"
                f.write(header + "\n")

        sys.stdout.write(line)
        sys.stdout.flush()

        with open("log.dat", "a") as f:
            f.writelines(line + "\n")

        with open("reaction_forces.dat", "a") as f:
            line = "%16e " % self.time_elapsed[None]
            for i in range(len(reactionForcesY)):
                line += "%16e " % reactionForcesY[i]
            f.writelines(line + "\n")


    def write_VTK(self):

        # this is quite a bit faster than writing VTK
        #filename = "snapshot_%06d" % (self.nSnapshot)
        #all_data = self.verts.to_numpy()
        #all_data.update(self.elems.to_numpy())
        #np.savez(filename, all_data)
        #self.nSnapshot += 1
        #return
        #print(all_data)


        points = self.verts.pos.to_numpy()
        points = np.column_stack((points, np.zeros(self.numVerts)))
        vel = self.verts.v.to_numpy()
        vel = np.column_stack((vel, np.zeros(len(vel))))
        vertexState = np.zeros(len(points), dtype=np.int32)

        elemState = self.elems.state.to_numpy()
        
        # intact elements
        idx = np.nonzero(elemState == ElementState.INTACT )
        a = self.elems.a.to_numpy()[idx]
        b = self.elems.b.to_numpy()[idx]
        c = self.elems.c.to_numpy()[idx]
        elems = np.column_stack((a, b, c))
        maxEig = self.elems.maxEig.to_numpy()[idx]
        stress = self.elems.stress.to_numpy()[idx]
        stress = stress[:,0,0]
        incenter_radius = self.elems.r.to_numpy()[idx]

        cells = {"triangle": elems}
        cell_data = {"maxEig": [maxEig], "stress": [stress], "r": [incenter_radius]}
        

        # failed elements are represented as dustballs
        idx_failed = np.nonzero(elemState == ElementState.FAILED)[0]
        nDustBalls = len(idx_failed)
        if nDustBalls > 0: # add dustballs te cells as VTK_VERTEX

            posDustballs = self.elems.dustPos.to_numpy()[idx_failed]
            posDustballs = np.column_stack((posDustballs, np.zeros(nDustBalls))) # append z=0 as we are 2D
            velDustballs = self.elems.dustVel.to_numpy()[idx_failed]
            velDustballs = np.column_stack((velDustballs, np.zeros(nDustBalls))) # append z=0 as we are 2D
            dustballState = np.ones(nDustBalls, dtype=np.int32)

            # elemsfailed is 1 1D array with the indices of the dustball positions
            # as the dustball positions are appended to the vertex positions,
            # we need to offset the indices by the number of vertices already in points
            lenPointsBefore = len(points)
            elems_failed = np.arange(nDustBalls) + lenPointsBefore
            elems_failed = elems_failed.reshape((nDustBalls, 1)) # need to up one axis for meshio

            points = np.vstack((points, posDustballs)) # add positions
            vel = np.vstack((vel, velDustballs)) # add velocities
            vertexState = np.append(vertexState, dustballState) # add some more random info
            
            incenter_radius = self.elems.r.to_numpy()[idx_failed]

            cells = {"triangle": elems, "vertex": elems_failed}
            zeros1D = np.zeros(nDustBalls, dtype=np.float32)
            zeros2D = np.zeros((nDustBalls,2,2), dtype=np.float32)
            cell_data["maxEig"].append(zeros1D)
            cell_data["stress"].append(zeros1D)
            cell_data["r"].append(incenter_radius)

        point_data = {"vel": vel, "state": vertexState}

        
        filename = "output/snapshot_%06d.xdmf" % (self.nSnapshot)
        meshio.write_points_cells(
            filename,
            points,
            cells,
            # Optionally provide extra data on points, cells, etc.
            point_data=point_data,
            cell_data=cell_data,
            # field_data=field_data
            )
        
        # also write the current simulation time to a file
        if self.nSnapshot == 0:
            mode = "w"
        else:
            mode = "a"
        with open("timestamps.csv", mode) as f:
            f.write(f"{self.nSnapshot};{self.time_elapsed[None]}\n")        
        
        self.nSnapshot += 1