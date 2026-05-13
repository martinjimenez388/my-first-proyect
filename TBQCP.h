#include "odes_solvers.h"
#include "C:\Users\Gigabyte\Documents\eigen3\unsupported\Eigen\KroneckerProduct"

using std::complex;
const int dim = 8;// The dimension of the Hilbert space.

Matrix2cd Pauli(int i)
{
    Matrix2cd sigma;
    if (i==0)
    {
        sigma.setIdentity();
    }
    else if (i==1)
    {
        sigma << complex<double>(0.,0.), complex<double> (1.,0.), complex<double>(1.,0.), complex<double>(0.,0.);
    }
    else if (i==2)
    {
        sigma << complex<double>(0.,0.), complex<double> (0.,-1.), complex<double>(0.,1.), complex<double>(0.,0.);
    }
    else if (i==3)
    {
        sigma << complex<double>(1.,0.), complex<double> (0.,0.), complex<double>(0.,0.), complex<double>(-1.,0.);
    }
    else
    {
        std::cout << "Indice erroneo\n";
        exit(1);
    }
    
    return sigma;    
}
Matrix2cd Id = Pauli(0), sigmax = Pauli(1), sigmay = Pauli(2), sigmaz = Pauli(3);
Matrix<complex<double>, 4, 4> BellOp = kroneckerProduct(sigmax, sigmax) + kroneckerProduct(sigmaz, sigmaz);
Matrix<complex<double>, dim, dim> H0 = -kroneckerProduct(Id, BellOp);
Matrix<complex<double>, dim, dim> H1 = -kroneckerProduct(BellOp, Id) ;

// The right-hand side of the Schodinger equation for the evolution operator
// or the quantum state.
template <class Ty>
Ty Solver<Ty>::rhs(const Ty &U, struct params &myp)
{
    Matrix<complex<double>, dim, dim> H;
    Ty dU;

    int elem = int(myp.t/myp.dt);
    // I use linear splines for the control fields.
    double E0 = myp.eps0[elem+1] - myp.eps0[elem];
    double E1 = myp.eps1[elem+1] - myp.eps1[elem];

    E0 = E0*(myp.t/myp.dt - elem) + myp.eps0[elem];
    E1 = E1*(myp.t/myp.dt - elem) + myp.eps1[elem];

    H = H0*E0 + H1*E1;
    dU = complex<double>(0., -1.) * H * U;

    return dU;
}
/*Fills the k-th row of a complex array of dim^2 columns with the elements of an Eigen::Matrix,
read by rows from top to bottom.*/
void Eigen2Array(const Matrix<complex<double>, dim, dim> &A, complex<double> *B, int k)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            B[dim*(dim*k + i) + j] = A(i,j);
        }        
    }    
}
/*Fills the elements of an Eigen::Matrix with the k-th row of a complex array of dim^2 columns.*/
void Array2Eigen(Matrix<complex<double>, dim, dim> &A, complex<double> *B, int k)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            A(i,j) = B[dim*(dim*k + i) + j];
        }        
    }    
}
/*Fills the k-th row of a complex array of dim columns with the elements of an Eigen::Vector.*/
void Vec2Array(const Vector<complex<double>, dim> &vec, complex<double> *B, int k)
{
    for (int i = 0; i < dim; i++)
    {
        B[dim*k + i] = vec(i);       
    }    
}
/*Fills the elements of an Eigen::Vector with the k-th row of a complex array of dim columns.*/
void Array2Vec(Vector<complex<double>, dim> &vec, complex<double> *B, int k)
{
    for (int i = 0; i < dim; i++)
    {
        vec(i) = B[dim*k + i];   
    }    
}