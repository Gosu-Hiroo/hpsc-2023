#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

int main(){
    int nx=41;
    int ny=41;
    int nt=500;
    int nit=50;
    double dx=2.0/(nx - 1);
    double dy=2.0/(ny - 1);
    double dt=.01;
    double rho=1;
    double nu=.02;

    std::vector<std::vector<double>> u(ny, std::vector<double>(nx, 0));
    std::vector<std::vector<double>> v(ny, std::vector<double>(nx, 0));
    std::vector<std::vector<double>> p(ny, std::vector<double>(nx, 0));
    std::vector<std::vector<double>> b(ny, std::vector<double>(nx, 0));

    for(int n=0;n<nt;++n){
        for(int j=1;j<ny - 1;++j){
            for(int i=1;i<nx - 1;++i){
                b[j][i]=rho*(1/dt*
                             ((u[j][i + 1] - u[j][i - 1])/(2*dx) + (v[j + 1][i] - v[j - 1][i])/(2*dy)) -
                             std::pow((u[j][i + 1] - u[j][i - 1])/(2*dx), 2) -
                             2*((u[j + 1][i] - u[j - 1][i])/(2*dy)*(v[j][i + 1] - v[j][i - 1])/(2*dx)) -
                             std::pow((v[j + 1][i] - v[j - 1][i])/(2*dy), 2));
            }
        }

        for(int it=0;it<nit;++it){
            std::vector<std::vector<double>> pn=p;

            for(int j=1;j<ny - 1;++j){
                for(int i=1;i<nx - 1;++i){
                    p[j][i]=(std::pow(dy, 2)*(pn[j][i + 1] + pn[j][i - 1]) +
                             std::pow(dx, 2)*(pn[j + 1][i] + pn[j - 1][i]) -
                             b[j][i]*std::pow(dx, 2)*std::pow(dy, 2))/
                            (2*(std::pow(dx, 2) + std::pow(dy, 2)));
                }
            }

            for(int i=0;i<ny;++i)p[i][nx - 1]=p[i][nx - 2], p[i][0]=p[i][1];
            for(int i=0;i<nx;++i)p[0][i]=p[1][i], p[ny - 1][i]=0;
        }

        std::vector<std::vector<double>> un=u;
        std::vector<std::vector<double>> vn=v;

        for(int j=1;j<ny - 1;++j){
            for(int i=1;i<nx - 1;++i){
                u[j][i]=un[j][i] - un[j][i]*dt/dx*(un[j][i] - un[j][i - 1]) -
                        vn[j][i]*dt/dy*(un[j][i] - un[j - 1][i]) -
                        dt/(2*rho*dx)*(p[j][i + 1] - p[j][i - 1]) +
                        nu*dt/std::pow(dx, 2)*(un[j][i + 1] - 2*un[j][i] + un[j][i - 1]) +
                        nu*dt/std::pow(dy, 2)*(un[j + 1][i] - 2*un[j][i] + un[j - 1][i]);

                v[j][i]=vn[j][i] - vn[j][i]*dt/dx*(vn[j][i] - vn[j][i - 1]) -
                        vn[j][i]*dt/dy*(vn[j][i] - vn[j - 1][i]) -
                        dt/(2*rho*dx)*(p[j + 1][i] - p[j - 1][i]) +
                        nu*dt/std::pow(dx, 2)*(vn[j][i + 1] - 2*vn[j][i] + vn[j][i - 1]) +
                        nu*dt/std::pow(dy, 2)*(vn[j + 1][i] - 2*vn[j][i] + vn[j - 1][i]);
            }
        }
        for(int i=0;i<ny;++i){
            u[i][0]=u[i][nx - 1]=0;
            v[i][0]=v[i][nx - 1]=0;
        }
        for(int i=0;i<nx;++i){
            u[0][i]=0;
            u[ny - 1][i]=1;
            v[ny - 1][i]=v[0][i]=0;
        }

        // data is saved and then visualized by a python program
        std::ofstream file_p("data/p_" + std::to_string(n) + ".txt");
        std::ofstream file_u("data/u_" + std::to_string(n) + ".txt");
        std::ofstream file_v("data/v_" + std::to_string(n) + ".txt");
        for(int j=0;j<ny;++j){
            for(int i=0;i<nx;++i){
                file_p << p[j][i] << " ";
                file_u << u[j][i] << " ";
                file_v << v[j][i] << " ";
            }
            file_p << std::endl;
            file_u << std::endl;
            file_v << std::endl;
        }
    }

    return 0;
}
