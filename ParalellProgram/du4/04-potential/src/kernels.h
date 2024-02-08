#ifndef CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H
#define CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <data.hpp>


/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception
{
protected:
	std::string mMessage;	///< Internal buffer where the message is kept.
	cudaError_t mStatus;

public:
	CudaError(cudaError_t status = cudaSuccess) : std::exception(), mStatus(status) {}
	CudaError(const char *msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	CudaError(const std::string &msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	virtual ~CudaError() throw() {}

	virtual const char* what() const throw()
	{
		return mMessage.c_str();
	}

	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	CudaError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};

/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char *srcFile, const char *errMsg = NULL)
{
	if (status != cudaSuccess) {
		throw (CudaError(status) << "CUDA Error (" << status << "): " << cudaGetErrorString(status) << "\n"
			<< "at " << srcFile << "[" << line << "]: " << errMsg);
	}
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)

using F = float;
using IDX_T = std::uint32_t;
using LEN_T = std::uint32_t;
using coord_t = F;		// Type of point coordinates.
using real_t = coord_t;	// Type of additional float parameters.
using index_t = IDX_T;
using length_t = LEN_T ;
using point_t = Point<coord_t>;
using edge_t = Edge<index_t>;

void setupForcesBeforeCompute(point_t *forces, const index_t num_points);
void computeRepulsiveForces(const point_t *points, point_t *forces, const index_t i, const F vertexRepulsion);
void addCompulsiveForce(const point_t *points, const edge_t *edges,
                        const index_t num_edges, const length_t *lengths,
                        const real_t edgeCompulsion, point_t *forces);

void computeVelocities(const point_t *forces, const real_t fact,
                       const real_t slowdown, point_t *velocities, const index_t num_vertices);

void updatePoints(point_t *points, const point_t *velocities, const real_t timeQuantum, const index_t num_points);

class CUDA_KERNELS
{
public:

    point_t* cu_points;
    edge_t* cu_edges;
    length_t* cu_lengths;

    point_t* cu_mForces;
    point_t* cu_mVelocities;

    index_t num_points;
    index_t num_edges;
    index_t num_iterations;
};

#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)


#endif
