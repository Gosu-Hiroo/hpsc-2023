#include <iostream>
#include <cmath>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define INDEX(i, j, nx) ((i)*(nx) + (j))

__global__ void calc_b(double *b, double *u, double *v, int nx, int ny, double dt, double dx, double dy, double rho){
    int i=blockIdx.x*blockDim.x + threadIdx.x;
    int j=blockIdx.y*blockDim.y + threadIdx.y;

    if(i>0 && i<nx - 1 && j>0 && j<ny - 1){
        b[INDEX(i, j, nx)]=rho*(1/dt*
                                ((u[INDEX(i, j + 1, nx)] - u[INDEX(i, j - 1, nx)])/(2*dx) +
                                 (v[INDEX(i + 1, j, nx)] - v[INDEX(i - 1, j, nx)])/(2*dy)) -
                                pow((u[INDEX(i, j + 1, nx)] - u[INDEX(i, j - 1, nx)])/(2*dx), 2) -
                                2*((u[INDEX(i + 1, j, nx)] - u[INDEX(i - 1, j, nx)])/(2*dy)*
                                   (v[INDEX(i, j + 1, nx)] - v[INDEX(i, j - 1, nx)])/(2*dx)) -
                                pow((v[INDEX(i + 1, j, nx)] - v[INDEX(i - 1, j, nx)])/(2*dy), 2));
    }
}

__global__ void calc_p(double *p, double *pn, double *b, int nx, int ny, double dx, double dy){
    int i=blockIdx.x*blockDim.x + threadIdx.x;
    int j=blockIdx.y*blockDim.y + threadIdx.y;

    if(i>0 && i<nx - 1 && j>0 && j<ny - 1){
        p[INDEX(i, j, nx)]=(pow(dy, 2)*(pn[INDEX(i, j + 1, nx)] + pn[INDEX(i, j - 1, nx)]) +
                            pow(dx, 2)*(pn[INDEX(i + 1, j, nx)] + pn[INDEX(i - 1, j, nx)]) -
                            b[INDEX(i, j, nx)]*pow(dx, 2)*pow(dy, 2))/
                           (2*(pow(dx, 2) + pow(dy, 2)));
    }
}

__global__ void
calc_uv(double *u, double *un, double *v, double *vn, double *p, int nx, int ny, double dt, double dx, double dy,
        double rho, double nu){
    int i=blockIdx.x*blockDim.x + threadIdx.x;
    int j=blockIdx.y*blockDim.y + threadIdx.y;

    if(i>0 && i<nx - 1 && j>0 && j<ny - 1){
        u[INDEX(i, j, nx)]=un[INDEX(i, j, nx)] -
                           un[INDEX(i, j, nx)]*dt/dx*(un[INDEX(i, j, nx)] - un[INDEX(i, j - 1, nx)]) -
                           vn[INDEX(i, j, nx)]*dt/dy*(un[INDEX(i, j, nx)] - un[INDEX(i - 1, j, nx)]) -
                           dt/(2*rho*dx)*(p[INDEX(i, j + 1, nx)] - p[INDEX(i, j - 1, nx)]) +
                           nu*dt/pow(dx, 2)*
                           (un[INDEX(i, j + 1, nx)] - 2*un[INDEX(i, j, nx)] + un[INDEX(i, j - 1, nx)]) +
                           nu*dt/pow(dy, 2)*(un[INDEX(i + 1, j, nx)] - 2*un[INDEX(i, j, nx)] + un[INDEX(i - 1, j, nx)]);

        v[INDEX(i, j, nx)]=vn[INDEX(i, j, nx)] -
                           vn[INDEX(i, j, nx)]*dt/dx*(vn[INDEX(i, j, nx)] - vn[INDEX(i, j - 1, nx)]) -
                           vn[INDEX(i, j, nx)]*dt/dy*(vn[INDEX(i, j, nx)] - vn[INDEX(i - 1, j, nx)]) -
                           dt/(2*rho*dx)*(p[INDEX(i + 1, j, nx)] - p[INDEX(i - 1, j, nx)]) +
                           nu*dt/pow(dx, 2)*
                           (vn[INDEX(i, j + 1, nx)] - 2*vn[INDEX(i, j, nx)] + vn[INDEX(i, j - 1, nx)]) +
                           nu*dt/pow(dy, 2)*(vn[INDEX(i + 1, j, nx)] - 2*vn[INDEX(i, j, nx)] + vn[INDEX(i - 1, j, nx)]);
    }
}

__global__ void edge_p(double *p, int nx, int ny){
    int i=blockIdx.x*blockDim.x + threadIdx.x;

    if(i<nx){
        p[INDEX(0, i, nx)]=p[INDEX(1, i, nx)];
        p[INDEX(ny - 1, i, nx)]=0;
        if(i<ny){
            p[INDEX(i, 0, nx)]=p[INDEX(i, 1, nx)];
            p[INDEX(i, nx - 1, nx)]=p[INDEX(i, nx - 2, nx)];
        }
    }
}

__global__ void edge_uv(double *u, double *v, int nx, int ny){
    int i=blockIdx.x*blockDim.x + threadIdx.x;

    if(i<ny){
        u[INDEX(i, 0, nx)]=0;
        u[INDEX(i, nx - 1, nx)]=0;
        v[INDEX(i, 0, nx)]=0;
        v[INDEX(i, nx - 1, nx)]=0;
    }

    if(i<nx){
        u[INDEX(0, i, nx)]=0;
        u[INDEX(ny - 1, i, nx)]=1;
        v[INDEX(ny - 1, i, nx)]=0;
        v[INDEX(0, i, nx)]=0;
    }
}

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

    double *u, *v, *p, *b, *pn, *un, *vn;

    cudaMallocManaged(&u, nx*ny*sizeof(double));
    cudaMallocManaged(&v, nx*ny*sizeof(double));
    cudaMallocManaged(&p, nx*ny*sizeof(double));
    cudaMallocManaged(&b, nx*ny*sizeof(double));
    cudaMallocManaged(&vn, nx*ny*sizeof(double));
    cudaMallocManaged(&pn, nx*ny*sizeof(double));
    cudaMallocManaged(&un, nx*ny*sizeof(double));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((nx + BLOCK_SIZE - 1)/BLOCK_SIZE, (ny + BLOCK_SIZE - 1)/BLOCK_SIZE);

    for(int n=0;n<nt;++n){
        calc_b<<<gridDim, blockDim>>>(b, u, v, nx, ny, dt, dx, dy, rho);
        cudaDeviceSynchronize();

        for(int it=0;it<nit;++it){
            cudaMemcpy(pn, p, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
            calc_p<<<gridDim, blockDim>>>(p, pn, b, nx, ny, dx, dy);
            cudaDeviceSynchronize();
            edge_p<<<gridDim, blockDim>>>(p, nx, ny);
            cudaDeviceSynchronize();
        }

        calc_uv<<<gridDim, blockDim>>>(u, un, v, vn, p, nx, ny, dt, dx, dy, rho, nu);
        cudaDeviceSynchronize();

        cudaMemcpy(un, u, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(vn, v, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
        edge_uv<<<gridDim, blockDim>>>(u, v, nx, ny);
        cudaDeviceSynchronize();

        // data is saved and then visualized by a python program
        std::ofstream file_p("data/p_" + std::to_string(n) + ".txt");
        std::ofstream file_u("data/u_" + std::to_string(n) + ".txt");
        std::ofstream file_v("data/v_" + std::to_string(n) + ".txt");
        for(int i=0;i<nx*ny;++i){
            if(i%nx==0){
                file_p << std::endl;
                file_u << std::endl;
                file_v << std::endl;
            }
            file_p << p[i] << " ";
            file_u << u[i] << " ";
            file_v << v[i] << " ";
        }
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);

    return 0;
}
