#include "kernels.h"
#include "math.h"

/*
 * Sample Kernel
 */
__global__ void my_kernel(float *src)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	src[idx] += 1.0f;
}


__global__ void cu_setupForcesBeforeCompute(point_t* forces, const index_t num_points)
{
    const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t stride = gridDim.x * blockDim.x;
    
    for (index_t i = tid; i < num_points; i += stride)
    {
        forces[i].x = forces[i].y = (real_t)0.0;
    }
}

void setupForcesBeforeCompute(point_t *forces, const index_t num_points){
    const int blockSize = 64;
    const int numBlocks = (num_points + blockSize - 1) / blockSize;
    cu_setupForcesBeforeCompute<<<numBlocks, blockSize>>>(forces, num_points);
}


__global__ void cu_computeRepulsiveForces(const point_t *points, point_t *forces, const index_t i, const F vertexRepulsion)
{
    const index_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= i) return;

    real_t dx = (real_t)points[i].x - (real_t)points[j].x;
    real_t dy = (real_t)points[i].y - (real_t)points[j].y;
    real_t sqLen = fmaxf(dx*dx + dy*dy, (real_t)0.0001);
    real_t fact = vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen)); // mul factor
    dx *= fact;
    dy *= fact;
    atomicAdd(&forces[i].x, dx);
    atomicAdd(&forces[i].y, dy);
    atomicAdd(&forces[j].x, -dx);
    atomicAdd(&forces[j].y, -dy);
}

void computeRepulsiveForces(const point_t *points, point_t* forces, const index_t i, const F vertexRepulsion){
    const int blockSize = 64;
    const int numBlocks = (i + blockSize - 1) / blockSize;
    cu_computeRepulsiveForces<<<numBlocks, blockSize>>>(points, forces, i, vertexRepulsion);
}


__global__ void cu_addCompulsiveForce(
    const point_t* points, const edge_t* edges, 
    const index_t num_edges, const length_t* lengths, 
    const real_t edgeCompulsion, point_t* forces)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_edges)
    {
        const index_t p1 = edges[i].p1;
        const index_t p2 = edges[i].p2;

        const real_t dx = (real_t)(points[p2].x - points[p1].x);
        const real_t dy = (real_t)(points[p2].y - points[p1].y);
        const real_t sqLen = dx * dx + dy * dy;
        const real_t fact = std::sqrt(sqLen) * edgeCompulsion / (real_t)(lengths[i]);

        const real_t fx = dx * fact;
        const real_t fy = dy * fact;

        atomicAdd(&(forces[p1].x), fx);
        atomicAdd(&(forces[p1].y), fy);
        atomicAdd(&(forces[p2].x), -fx);
        atomicAdd(&(forces[p2].y), -fy);
    }
}


void addCompulsiveForce(const point_t* points, const edge_t* edges, 
    const index_t num_edges, const length_t* lengths, 
    const real_t edgeCompulsion, point_t* forces)
{   
    const int blockSize = 64;
    const int numBlocks = (num_edges + blockSize - 1) / blockSize;
    cu_addCompulsiveForce<<<numBlocks, blockSize>>>(points, edges, num_edges, lengths, edgeCompulsion, forces);
}


__global__ void cu_computeVelocities(const point_t* forces, const real_t fact, const real_t slowdown, 
        point_t* velocities, const index_t num_points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_points) {
        velocities[i].x = ( [i].x + (real_t)forces[i].x * fact) * slowdown;
        velocities[i].y = (velocities[i].y + (real_t)forces[i].y * fact) * slowdown;
    }
}

void computeVelocities(const point_t* forces, const real_t fact, 
        const real_t slowdown, point_t* velocities, const index_t num_points){
    const int blockSize = 64;
    const int numBlocks = (num_points + blockSize - 1) / blockSize;

    cu_computeVelocities<<<numBlocks, blockSize>>>(forces, fact, slowdown, velocities, num_points);
}