#%% #!/usr/bin/env python3

Description = """
Script used to perform theorical computation
of hyperelastic strain and stress for uniaxial tensile test
"""

__author__ = ['Mathieu Simon']
__date_created__ = '27-01-2025'
__date__ = '27-01-2025'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import inspect
import argparse
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display


#%% Main

def Main():

    # Identity tensor
    I = sp.eye(3)

    # Direction vector
    e1 = I[:,0]
    e2 = I[:,1]
    e3 = I[:,2]

    # Dilatation coefficients (Poisson's ratio:https://en.wikipedia.org/wiki/Poisson%27s_ratio - Length change)
    u, Nu, Mu   = sp.symbols(r'u \nu \mu', positive=True)

    LambdaH = 1             # Coefficient for homogeneous dilatation
    LambdaX = u **(-Nu)     # Coefficient for lengthening in e1 direction (here <1)
    LambdaY = u **(-Nu)     # Coefficient for lengthening in e2 direction (here <1)
    LambdaZ = u             # Coefficient for lengthening in e3 direction (here >1)

    # Deformation matrices
    U = I + (LambdaX-1) * np.outer(e1,e1) + (LambdaY-1) * np.outer(e2,e2) + (LambdaZ-1) * np.outer(e3,e3)

    # Gradient of the deformation
    F = (LambdaH-1) * I + U

    # Right Cauchy-Green strain tensor
    C = sp.transpose(F) * F

    # Eigenvalues of Right Cauchy-Green strain tensor
    Lambda1, Lambda2, Lambda3 = sp.symbols(r'\lambda_1 \lambda_2 \lambda_3')

    # Invariants
    J, I1, I2 = sp.symbols(r'J I_1 I_2')
    JFunction = Lambda1*Lambda2*Lambda3
    I1Function = Lambda1**2+Lambda2**2+Lambda3**2
    I2Function = Lambda1**2*Lambda2**2 + Lambda2**2*Lambda3**2 + Lambda3**2*Lambda1**2

    # Hyperelastic models (compressible)

    # Neo-Hookean
    C1, D1 = sp.symbols(r'C_{1} D_{1}', positive=True)
    C1Function  = Mu / 2
    D1Function  = Mu*Nu / (1 - 2*Nu)
    Psi_NH = C1 * (J**sp.Rational(-2,3)*I1 - 3) + D1 * (J-1)**2

    Psis = [Psi_NH]
    Psi  = Psis[0]
    display(Psi)

    # Substitute eigenvalues and material parameters
    Psi = Psi.subs({J:JFunction,I1:I1Function})
    display(Psi)
    Psi = Psi.subs({C1:C1Function,D1:D1Function})
    display(Psi)

    # Derivative with respect to Lambdas (https://en.wikipedia.org/wiki/Hyperelastic_material: compressible isotropic hyperelastic material)
    T1 = Lambda1*sp.Derivative(Psi, Lambda1)/(Lambda1*Lambda2*Lambda3)
    T1 = T1.doit()

    T2 = Lambda2*sp.Derivative(Psi, Lambda2)/(Lambda1*Lambda2*Lambda3)
    T2 = T2.doit()

    T3 = Lambda3*sp.Derivative(Psi, Lambda3)/(Lambda1*Lambda2*Lambda3)
    T3 = T3.doit()

    T = T1 * np.outer(e1,e1) + T2 *np.outer(e2,e2) + T3 * np.outer(e3,e3)     # Add the pressure p for incompressibility
    T = sp.Matrix(T).doit()
    display(T)

    # Replace Eingenvalues
    Lambdas = C.eigenvals()
    Lambdas = list(Lambdas)
    T = T.subs({Lambda1:Lambdas[0], Lambda2:Lambdas[0], Lambda3:Lambdas[1]})
    display(T)

    # Pure tensile test: T11 and T22 are null
    T = T - T[0,0] * I
    T = sp.simplify(T)
    display(T)

    # Other stresses
    J = sp.det(F)                                 # Volume change
    P = J * T * F.inv().transpose()               # Nominal stress
    S = J * F.inv() * T * F.inv().transpose()     # Material stress
    display(S)

    # Define axial responses
    NH = sp.lambdify((Nu, Mu, u), P[2,2], 'numpy')

    # Plot Results
    NuV = 0.4
    MuV = 1

    Xmin = 0.45
    Xmax = 2.55
    Delta = 0.01
    U33 = np.arange(Xmin,Xmax+Delta,Delta)

    Figure, Axis = plt.subplots(1,1)
    Axis.plot(U33, NH(NuV, MuV, U33),  color = 'k', linestyle = '-', label='Neo-Hookean')
    # ax.plot(U33, MR(Nu,U33),  color = 'g', linestyle = '-', label='Mooney-Rivlin')
    # ax.plot(U33, Gn(Nu,U33),  color = 'b', linestyle = '-', label='Gent')
    # ax.plot(U33, Dm(Nu,U33),  color = 'c', linestyle = '-', label='Demiray')
    # ax.plot(U33, Og(Nu,U33),  color = 'r', linestyle = '-', label='Ogden')
    Axis.set_xlabel('Stretch ratio (-)')
    Axis.set_ylabel('Stress (kPa)')
    plt.legend(loc='upper left')
    plt.show()

    # Store function into python file
    Function = inspect.getsource(NH)
    Function = Function.replace('_lambdifygenerated', 'NeoHookean')
    Function = Function.replace('Dummy_81','Nu')
    Function = Function.replace('Dummy_82','Mu')
    with open('Function.py','w') as F:
        F.write(Function)

    return

if __name__ == '__main__':
    
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main()

        