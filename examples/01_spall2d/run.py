import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../../2D')) # insert the path 

from fem2d import FEM

runProperties = {
    "runDuration": 1.0e-3,
    "dtCFL": 0.35,
    "damping": 0.02,
    "intervalVTU": -1.0e-6,
    "intervalRender": 1.0e-5,
    "boxPadTop": 0.05,
    "boxPadRight": 0.20,
    "boxPadBot": 0.05,
    "boxPadLeft": 0.05,
}

materialProperties = {
    0: {
            "E": 200.0,
            "nu": 0.0,
            "rho": 1.2e-6,
            "epsfail": 1.0,
            "Gf": 1.0e5
        },
    1: {
            "E": 200.0,
            "nu": 0.0,
            "rho": 1.2e-6,
            "epsfail": 0.02,
            "Gf": 1.0e5
        }}

initialVelocities = {
    0: [1000.0, 0.0],
    1: [0.0, 0.0],
}

prescribedVelocities = {
    "Y_flag" : [],
    "Y_value" : [],
}

renderParams = {
    "renderElements": True,
    "renderElementOutlines": False,
    "renderFreeVertices": False,
    "renderDustballs": True,
    "renderVertices": False,
    "renderBoundaryParticles": False,
    "renderNodeSets": False
}

contactProperties = {
    "contactStiffness": 1.0, # dimensionless scale factor for contact stiffness
    "restitutionCoefficient": 0.1,
    "includeDustBalls": True,
    "contactDamping": 1.0
}

fem = FEM(materialProperties, initialVelocities, renderParams, contactProperties, runProperties, prescribedVelocities)

