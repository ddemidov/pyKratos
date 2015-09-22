from __future__ import print_function, absolute_import, division 
from numpy import *
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab
from pyKratos.variables import PRESSURE
import pyamgcl as amg

class BuilderAndSolver:
    use_sparse_matrices = True

    '''ATTENTION!!
    this builder and solver assumes elements to be written IN RESIDUAL FORM and hence
    solves FOR A CORRECTION Dx'''

    def __init__(self, model_part, scheme):
        self.scheme = scheme
        self.model_part = model_part
        self.dofset = set()
        self.dirichlet_dof = set()

    def SetupDofSet(self):
        '''this function shapes the system to be built'''

        # start by iterating over all the elements and obtaining the list of
        # dofs
        aux = set()
        for elem in self.model_part.ElementIterators():
            unknowns = elem.GetDofList()

            for aaa in unknowns:
                aux.add(aaa)

        self.dofset = sorted(aux)

        # for dof in self.dofset:
            # print dof.node.Id, " ",dof.variable

        # assign an equation id
        counter = 0
        for dof in self.dofset:
            dof.SetEquationId(counter)
            counter += 1

            if(dof.IsFixed()):
                self.dirichlet_dof.add(dof)

    def SetupSystem(self, A, dx, b):
        ndof = len(self.dofset)

        # allocate systme vectors
        b = zeros(ndof)
        dx = zeros(ndof)

        # allocate system matrix
        if(self.use_sparse_matrices == False):  # dense case
            A = zeros((ndof, ndof))
        else:  # allocate non zeros and transofrm to csr
            A = sparse.dok_matrix((ndof, ndof))
            for elem in self.model_part.ElementIterators():
                # get non zero positions
                equation_id = self.scheme.EquationId(elem)
                for i in range(0, len(equation_id)):
                    eq_i = equation_id[i]
                    for j in range(0, len(equation_id)):
                        eq_j = equation_id[j]
                        A[eq_i,
                            eq_j] = 1.0  # set it to 1 to ensure it is converted well
                        # problem here is that in converting zero entries are
                        # discarded
            A = A.tocsr()

        return [A, dx, b]

    # this function sets to
    def SetToZero(self, A, dx, b):
        ndof = len(self.dofset)

        if(self.use_sparse_matrices == False):
            # allocating a dense matrix. This should be definitely improved
            A = zeros((ndof, ndof))
            b = zeros(ndof)
            dx = zeros(ndof)
        else:
            # print A.todense()
            A = A.multiply(
                0.0)  # only way i found to set to zero is to multiply by zero
            b = zeros(ndof)
            dx = zeros(ndof)
        return [A, dx, b]

    def ApplyDirichlet(self, A, dx, b):
        ndof = A.shape[0]
        if(self.use_sparse_matrices == False):
            for dof in self.dirichlet_dof:
                fixed_eqn = dof.GetEquationId()
                for i in range(0, ndof):
                    A[fixed_eqn, i] = 0.0
                    A[i, fixed_eqn] = 0.0
                A[fixed_eqn, fixed_eqn] = 1.0
                b[fixed_eqn] = 0.0  # note that this is zero since we assume residual form!
        else:
            # expensive loop: exactly set to 1 the diagonal
            # could be done cheaper, but i want to guarantee accuracy
            aux = ones(ndof, dtype=bool)
            for dof in self.dirichlet_dof:
                eq_id = dof.GetEquationId()
                aux[eq_id] = False


            ij = A.nonzero()
            for i, j in zip(ij[0], ij[1]):
                if(aux[i] == False or aux[j] == False):
                    A[i, j] = 0.0

            for dof in self.dirichlet_dof:
                eq_id = dof.GetEquationId()
                A[eq_id, eq_id] = 1.0
                b[eq_id] = 0.0

        return [A, dx, b]
    
    def Build(self, A, dx, b):
        A, dx, b = self.SetToZero(A, dx, b)

        for elem in self.model_part.ElementIterators():
            # find where to assemble
            equation_id = self.scheme.EquationId(elem)

            # compute local contribution to the stiffness matrix
            [lhs, rhs] = self.scheme.CalculateLocalSystem(elem)

            # assembly to the matrix
            for i in range(0, len(equation_id)):
                eq_i = equation_id[i]
                b[eq_i] += rhs[i]
                for j in range(0, len(equation_id)):
                    eq_j = equation_id[j]
                    A[eq_i, eq_j] += lhs[i, j]

        for cond in self.model_part.ConditionIterators():
            # find where to assemble
            equation_id = self.scheme.EquationId(cond)

            # compute local contribution to the stiffness matrix
            [lhs, rhs] = self.scheme.CalculateLocalSystem(cond)

            # assembly to the matrix
            for i in range(0, len(equation_id)):
                eq_i = equation_id[i]
                b[eq_i] += rhs[i]
                for j in range(0, len(equation_id)):
                    eq_j = equation_id[j]
                    A[eq_i, eq_j] += lhs[i, j]
        return [A, dx, b]

    def BuildAndSolve(self, A, dx, b):
        A, dx, b = self.Build(A, dx, b)

        A, dx, b = self.ApplyDirichlet(A, dx, b)

        # print A
        if(self.use_sparse_matrices == False):
            dx = linalg.solve(A, b)
        else:
            n = A.shape[0]

            mp = -1 * ones(n)
            ms = -1 * ones(n)

            np = 0
            ns = 0

            for i,dof in enumerate(self.dofset):
                if dof.variable == PRESSURE:
                    mp[i] = np
                    np += 1
                else:
                    ms[i] = ns
                    ns += 1

            App = sparse.lil_matrix((np,np), dtype=float64)
            Ass = sparse.lil_matrix((ns,ns), dtype=float64)
            Aps = sparse.lil_matrix((np,ns), dtype=float64)
            Asp = sparse.lil_matrix((ns,np), dtype=float64)

            ij = A.nonzero()
            for i, j in zip(ij[0], ij[1]):
                if ms[i] >= 0 and ms[j] >= 0:
                    Ass[ms[i],ms[j]] = A[i,j]
                elif ms[i] >= 0 and mp[j] >= 0:
                    Asp[ms[i],mp[j]] = A[i,j]
                elif mp[i] >= 0 and ms[j] >= 0:
                    Aps[mp[i],ms[j]] = A[i,j]
                elif mp[i] >= 0 and mp[j] >= 0:
                    App[mp[i],mp[j]] = A[i,j]

            Dss = sparse.spdiags((Ass.diagonal()**-1), [0], ns, ns)
            Aps_Dss = Aps.dot(Dss)
            Ap = App - Aps_Dss.dot(Asp)

            Pp  = amg.make_preconditioner(Ap, prm={"coarse_enough" : 100})
            ILU = sparse.linalg.spilu(A, fill_factor=2)

            def applyM(b):
                if True:
                    bp = zeros(np)
                    bs = zeros(ns)

                    for i in range(n):
                        if ms[i] >= 0:
                            bs[ms[i]] = b[i]
                        else:
                            bp[mp[i]] = b[i]

                    rp = bp - Aps_Dss * bs
                    xp = Pp(rp)

                    dx = zeros(n)
                    for i in range(n):
                        if mp[i] >= 0:
                            dx[i] = xp[mp[i]]

                    newb = b - A * dx

                    x = ILU.solve(newb)

                    return x + dx
                else:
                    return ILU.solve(b)

            M = LinearOperator((n,n), matvec=applyM)

            numiter = [0]

            def callback(x):
                numiter[0] += 1
                res = linalg.norm(b - A * x) / linalg.norm(b)
                print("iter: %s, res: %s" % (numiter[0], res))

            dx,info = bicgstab(A, b, M=M, callback=callback)

        return [A, dx, b]
