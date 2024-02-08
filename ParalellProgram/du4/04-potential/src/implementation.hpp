#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP


#include <interface.hpp>
#include <data.hpp>

#include "kernels.h"

#include <cuda_runtime.h>

#include <iostream>




/*
 * Final implementation of the tested program.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T>
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;
    
private:

	void computeForces(std::vector<point_t> &points)
	{
        setupForcesBeforeCompute(kernels.cu_mForces, kernels.num_points);
        
        for (index_t i = 1; i < kernels.num_points; ++i)
        {
            computeRepulsiveForces(kernels.cu_points, kernels.cu_mForces, i, this->mParams.vertexRepulsion);
        }

        addCompulsiveForce(kernels.cu_points, kernels.cu_edges, 
            kernels.num_edges, kernels.cu_lengths, 
            this->mParams.edgeCompulsion, kernels.cu_mForces);
    
        // Check for CUDA exceptions
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("CUDA error compute forces: %s\n", cudaGetErrorString(error));
        }
    }
    
    void updateVelocities()
	{
		real_t fact = this->mParams.timeQuantum / this->mParams.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
        computeVelocities(kernels.cu_mForces, fact, this->mParams.slowdown, 
            kernels.cu_mVelocities, kernels.num_points);

            // Check for CUDA exceptions
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("CUDA error update velocities: %s\n", cudaGetErrorString(error));
        }
    }

    CUDA_KERNELS kernels;
public:
    virtual void initialize(index_t points, 
        const std::vector<edge_t>& edges, 
        const std::vector<length_t>& lengths, 
        index_t iterations)
    {
        // Allocate device memory for points, edges, and lengths arrays
        size_t points_size = points * sizeof(point_t);
        point_t* cu_points;
        cudaMalloc((void**)&cu_points, points_size);
        cudaMemset(cu_points, 0, points_size);

        point_t* cu_mForces;
        cudaMalloc((void**)&cu_mForces, points_size);
        cudaMemset(cu_mForces, 0, points_size);
        
        point_t* cu_mVelocities;
        cudaMalloc((void**)&cu_mVelocities, points_size);
        cudaMemset(cu_mVelocities, 0, points_size);
        
        size_t edges_size = edges.size() * sizeof(edge_t);
        edge_t* cu_edges;
        cudaMalloc((void**)&cu_edges, edges_size);

        size_t lengths_size = lengths.size() * sizeof(length_t);
        length_t* cu_lengths;
        cudaMalloc((void**)&cu_lengths, lengths_size);

        // Copy edges, and lengths data from host to device
        cudaMemcpy(cu_edges, edges.data(), edges_size, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_lengths, lengths.data(), lengths_size, cudaMemcpyHostToDevice);

        // Store device pointers and other state in class members
        kernels.cu_edges = cu_edges;
        kernels.cu_lengths = cu_lengths;
        kernels.cu_mForces = cu_mForces;
        kernels.cu_mVelocities = cu_mVelocities;
        kernels.num_points = points;
        kernels.num_edges = edges.size();
        kernels.num_iterations = iterations;

        // Check for CUDA exceptions
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error initialize: %s\n", cudaGetErrorString(error));
    }
    }




	virtual void iteration(std::vector<point_t> &points)
	{
        if (points.size() != kernels.num_points)
			throw (bpp::RuntimeError() << "Cannot compute next version of point positions."
				<< "Current model uses " << kernels.num_points << " points, but the given buffer has " << points.size() << " points.");

		/*
		 * Perform one iteration of the simulation and update positions of the points.
		 */
        computeForces(points);
        updateVelocities();
        std::cout << std::endl << "num points: " << kernels.num_points << std::endl;
        cudaMemcpy(points.data(), kernels.cu_points, kernels.num_points * sizeof(point_t), cudaMemcpyDeviceToHost);        
        std::cout << std::endl << "points size: " << points.size() << std::endl;
        // Check for CUDA exceptions
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error iteration: %s\n", cudaGetErrorString(error));
    }
    }

    virtual void getVelocities(std::vector<point_t> &velocities)
	{
		/*
		 * Retrieve the velocities buffer from the GPU.
		 * This operation is for vreification only and it does not have to be efficient.
		 */
        cudaMemcpy(velocities.data(), kernels.cu_mVelocities, kernels.num_points * sizeof(point_t), cudaMemcpyDeviceToHost);        
	}
};


#endif
