# A bare-bones,traditional infinite DMRG implementation for the 
#    transverse Ising model.
# A more sophisticated version would be fully MPS based, allow for 
#    finite system size, expoit quantum symmetries, and calculate 
#    observables, correlations, entanglement, etc.
# iTensor is a nice library with all these capabilities (http://itensor.org/)
#
# Hamiltonian:
#    The Hamiltonian of the transverse Ising model is given below:
#    H = -\sum_{i} J \sigma_{z,i}\sigma_{z,i+1} + h \sigma_{x,i}
# 
#    Since DMRG is quasi-1D, we must employ open boundaries

import numpy as np
import sys
from collections import namedtuple
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import linalg
from scipy.sparse.linalg import eigsh
from scipy.sparse import kron
from scipy.sparse.linalg import svds as svd
from scipy.sparse import identity as ids

# Define relevant local operators for single-site spin system.
# Sparsity allows for efficient kronecker products when constructing 
#    many-body operators.
sigma_0 = sparse.coo_matrix(np.array([[1.0, 0.0],[0.0, 1.0]]))
sigma_x = sparse.coo_matrix(np.array([[0.0, 1.0],[1.0, 0.0]]))
sigma_y = sparse.coo_matrix(np.array([[0.0, -1.0j],[1.0j, 0.0]]))
sigma_z = sparse.coo_matrix(np.array([[1.0, 0.0],[0.0, -1.0]]))

# Symmetrically grow MPS and truncate via SVD
def GrowSparse(H,params,trans):
   sys_dim   = H.A.shape[0]
   dim       = sys_dim*params['loc_dim']
   U         = trans.A[len(trans.A)-1]
   Vh        = trans.B[len(trans.B)-1]
   old_dim   = U.shape[0]
   H_Anew    = kron(H.A,sigma_0)
   H_Bnew    = kron(sigma_0,H.B)
   sz_Utrans = U.conj().T.dot(kron(ids(old_dim//params['loc_dim'],dtype=np.float),sigma_z).dot(U))
   H_Adot    = -params['J']*kron(sz_Utrans,sigma_z)
   sz_Vtrans = Vh.conj().dot(kron(sigma_z,ids(old_dim//params['loc_dim'],dtype=np.float)).dot(Vh.T))
   H_dotB    = -params['J']*kron(sigma_z,sz_Vtrans)
   H_dotdot  = -params['J']*kron(kron(kron(ids(sys_dim,dtype=np.float),sigma_z),sigma_z),ids(sys_dim,dtype=np.float))
   H_dotdotA = -params['h']*kron(ids(sys_dim,dtype=np.float),sigma_x)
   H_dotdotB = -params['h']*kron(sigma_x,ids(sys_dim,dtype=np.float))
   H_super   = kron(H_Anew,ids(dim,dtype=np.float)) + \
               kron(ids(dim,dtype=np.float),H_Bnew) + \
               kron(H_Adot,ids(dim,dtype=np.float)) + \
               kron(ids(dim,dtype=np.float),H_dotB) + \
               H_dotdot + \
               kron(H_dotdotA,ids(dim,dtype=np.float)) + \
               kron(ids(dim,dtype=np.float),H_dotdotB)
   eigval, psi = eigsh(H_super, k=1,tol=1.e-8,which = 'SA')
   energy.append(eigval.tolist()[0])

   # Since we have A <-> B symmetry, we may reshape psi here instead of 
   #    looking at tr_A |psi><psi|
   rho_dim = int(np.sqrt(psi.shape[0]))
   if (np.abs(rho_dim - np.sqrt(psi.shape[0])) > 1e-10): # Check int sqrt
      print ('Error: Density matrix dimension not int')
      quit()
   psi_ij   = np.reshape(psi, (rho_dim,rho_dim), order='C') 
   psi_ijs  = sparse.coo_matrix(psi_ij)
   num_SV   = min(params['maxM'],dim-1)
   U, s, Vh = svd(psi_ijs,k=num_SV)
   if sys_dim >= params['maxM']:
      U = U[:,0:params['maxM']]
      Vh = Vh[0:params['maxM'],:]
   trans.A.append(U)
   trans.B.append(Vh)
   H_Amod = H_Anew + H_Adot + H_dotdotA
   H_Bmod = H_Bnew + H_dotB + H_dotdotB
   H_new = Ham(U.conj().T.dot(H_Amod.dot(U)),Vh.conj().dot(H_Bmod.dot(Vh.T)))
   return H_new

# Model parameters:
J       = 1.0    # spin-spin coupling strength
h       = 1.0    # transverse field stregth
loc_dim = 2      # local single-site Hilbert space dimension (2 for spin-half system)
maxM    = 20     # max number of states kept in truncation step
it_num  = 100   # number of growth iterations

params = {'J': J, 'h': h, 'loc_dim': loc_dim, 'maxM': maxM}

# INITIALIZATION
#   We use the notation common in the DMRG literature that 
#   A corresponds to the left subsystem and B corresponds to the right
Ham = namedtuple('Ham',['A','B'])
H_A_init = -params['h']*sigma_x
H_B_init = -params['h']*sigma_x
ham = Ham(H_A_init,H_B_init)
Trans = namedtuple('Trans',['A','B']) # Collection of reduced basis transformations for representing H
basis_trans = Trans([sigma_0],[sigma_0])
energy = []

for i in range(0,it_num):
  #print ('Iteration ', i)
  ham = GrowSparse(ham,params,basis_trans)
  print ('Iteration ', i, ':', '    Energy/N = ', energy[i]/(2*(i+1)+2), sep='')

# The exact result for this model is known.
# You can Jordan-Wigner transform and write the ground state energy as an elliptic 
#   integral of the second kind.
#   cf Pfeuty, Pierre. "The One-Dimensional Ising Model with a Transverse Field".
#   Annals of Physics 57, 79-90 (1970)
# Since DMRG is variational, you should never dip below this energy.

from scipy.special import ellipeinc
PI = np.pi

print('\nExact Energy/N =', -params['J']*2/PI*(1+params['J']/params['h'])*ellipeinc(PI/2,4*(params['J']/params['h'])/(1+params['J']/params['h'])**2))
