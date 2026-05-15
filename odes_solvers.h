/*Simple ODE's solvers. The functions defined here get a vector or matrix x, a struct with the system parameters
(including time if explicitly appears), the right-hand side function f (depending on the system parameters)
of dx/dt = f(x), and the time step dt. They return x evolved one time step with one of these methods:
Euler and Runge-Kutta 4.
*/

#include <iostream>
#include <functional>
#include "C:\Users\Gigabyte\Documents\eigen3\Eigen\Dense"
#include "C:\Users\Gigabyte\Documents\eigen3\Eigen\Core"

using std::function;
using namespace Eigen;

// It must be adapted depending on your specific system.
struct params
{
    double t;
    double dt;//Time step lenght for the integrator
    double Tf;
    int dir;//Set it to 1 or -1 to evolve forward or backward
    double *eps0;
    double *eps1;
};


template <class Ty>
class Solver {
  public:
    Ty rhs (const Ty &, struct params &);
    Ty Euler (const Ty &, struct params &, const function<Ty (const Ty & , struct params &)> &);
    Ty Heun (const Ty &, struct params &, const function<Ty (const Ty & , struct params &)> &);
    Ty RK4(const Ty &, struct params &, const function<Ty (const Ty &, struct params &)> &);
};

template <class Ty>
Ty Solver<Ty>::Euler(const Ty &x, struct params &myp, const function<Ty (const Ty &, struct params &)> &f){
    Ty y;

    y = x + myp.dir*myp.dt * f(x, myp);
    myp.t += myp.dir*myp.dt;

    return y;
}

template <class Ty>
Ty Solver<Ty>::Heun(const Ty &x, struct params &myp, const function<Ty (const Ty &, struct params &)> &f){
    Ty y, k1, k2;
    double tst = myp.dir*myp.dt;

    k1 = tst* f(x, myp)
    myp.t += tst;
    k2 = tst* f(x + k1, myp);
    y = x + 0.5*(k1 + k2);

    return res;
}

template <class Ty>
Ty Solver<Ty>::RK4(const Ty &x, struct params &myp, const function<Ty (const Ty &, struct params &)> &f){
    Ty k1, k2, k3, k4, y;
    double tst = myp.dir*myp.dt;

    k1 = tst * f(x, myp);
    myp.t += 0.5*tst;
    k2 = tst * f(x + 0.5*k1, myp);
    k3 = tst * f(x + 0.5*k2, myp);
    myp.t += 0.5*tst;
    k4 = tst * f(x + k3, myp);

    y = x + (k1 + 2.*k2 + 2.*k3 + k4)/6.;

    return y;
}
