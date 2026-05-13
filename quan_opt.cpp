#include "TBQCP.h"
#include <cmath>
#include <fstream>

using std::cout;
using std::endl;

double guess_ctrl(struct params &myp)
{
    double y = 0.;
    double t0 = myp.Tf/30.;

    if (myp.t<t0)  y = sin(M_PI_2*myp.t/t0);
    else if (myp.t>myp.Tf-t0) y = sin(M_PI_2*(myp.t-myp.Tf)/t0);
    else y = 1.0;
    
    return y*y;
}

int main()
{
    Solver <Vector<complex<double>, dim>> Vevolve;//Vector time evolution
    Solver <Matrix<complex<double>, dim, dim>> Mevolve;//Matrix time evolution
    Matrix<complex<double>, dim, dim> U;//The evolution operator
    Matrix<complex<double>, dim, dim> OT;//The target gate
    Vector<complex<double>, dim> psi_target, psi, psi_init;
    Vector2cd q0;//Qubit state we want to teleport
    Vector<complex<double>, 4> bell_state;
    struct params myp;
    complex<double> *save_Ot, *save_psi;//Saves values of the state or the propagator through the time grid.
    double eta = 1.e-2;//The correction rate of the control fields.
    int niter = 20;//The number of iterations of the algorithm.
    
    myp.Tf = 1.2;
    myp.dt = 2.e-3;
    int ntsteps = std::round(myp.Tf/myp.dt);

    q0(0)=M_SQRT1_2; q0(1)=M_SQRT1_2;
    bell_state.setZero(); bell_state(0)=M_SQRT1_2; bell_state(3)=M_SQRT1_2;
    psi_target = kroneckerProduct(bell_state, q0);
    psi_init = kroneckerProduct(q0, bell_state);
    OT = psi_target*psi_target.adjoint();

    save_psi = new complex<double>[(ntsteps+1)*dim];
    save_Ot = new complex<double>[(ntsteps+1)*dim*dim];

    /*Set the guess control fields.*/
    myp.eps0 = new double[ntsteps+1];
    myp.eps1 = new double[ntsteps+1];
    for (int i = 0; i <= ntsteps; i++)
    {
        myp.t = i*myp.dt;
        myp.eps1[i] = guess_ctrl(myp);
        myp.eps0[i] = 1. - myp.eps1[i];
    }
    //Main TBQCP iteration loop
    for (int k = 0; k < niter; k++)
    {
        /*Backward-evolved time-invariant operator from the target gate*/
        myp.dir = -1;
        myp.t = myp.Tf;
        U = OT;
        Eigen2Array(OT, save_Ot, ntsteps);
        for (int i = ntsteps-1; i >= 0; i--)
        {
            U = Mevolve.RK4(U, myp, [&](const auto &u, params &p) { return Mevolve.rhs(u, p); });
            Eigen2Array(U*OT*U.adjoint(), save_Ot, i);
        }

        /*Forward-evolved quantum state*/
        myp.dir = 1;
        myp.t = 0.;
        psi = psi_init;
        Vec2Array(psi, save_psi, 0);
        for (int i = 1; i <= ntsteps; i++)
        {
            psi = Vevolve.RK4(psi, myp, [&](const auto &u, params &p) { return Vevolve.rhs(u, p); });
            Vec2Array(psi, save_psi, i);
        }

        /*Compute the corrections to the control fields, and update them.*/
        for (int i = 0; i <= ntsteps; i++)
        {
            Array2Eigen(U, save_Ot, i);
            Array2Vec(psi, save_psi, i);

            myp.eps0[i] += 2.*eta*(psi.adjoint()*U*H0*psi).value().imag();
            myp.eps1[i] += 2.*eta*(psi.adjoint()*U*H1*psi).value().imag();
        }
    }

    myp.dir = 1;
    myp.t = 0.;
    psi = psi_init;
    std::ofstream arch("corr_control_fields.csv", std::ios::out);
    arch << "Time,Eps0,Eps1" << endl;
    for (int i = 0; i <= ntsteps; i++)
    {
        arch << myp.t << ",";
        arch << myp.eps0[i] << ",";
        arch << myp.eps1[i] << ",";
        arch << pow(abs(psi_target.dot(psi)), 2) << endl;
        psi = Vevolve.RK4(psi, myp, [&](const auto &u, params &p) { return Vevolve.rhs(u, p); });
    }
    arch.close();

    delete [] myp.eps0;
    delete [] myp.eps1;
    delete [] save_psi;
    delete [] save_Ot;

    return 0;
}
