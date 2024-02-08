#ifndef CUDA_POTENTIAL_INTERNAL_SERIAL_HPP
#define CUDA_POTENTIAL_INTERNAL_SERIAL_HPP

#include <data.hpp>
#include <exception.hpp>

#include <vector>
#include <algorithm>
#include <cmath>

template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class SerialSimulator
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:
	const std::vector<edge_t> &mEdges;		///< Reference to the graph edges.
	const std::vector<length_t> &mLengths;	///< Reference to the graph lengths of the edges.
	std::vector<point_t> mVelocities;			///< Point velocity vectors.
	std::vector<point_t> mForces;				///< Preallocated buffer for force vectors.
	ModelParameters<F> mParams;

	void addRepulsiveForce(const std::vector<point_t> &points, index_t p1, index_t p2, std::vector<point_t> &forces)
	{
		real_t dx = (real_t)points[p1].x - (real_t)points[p2].x;
		real_t dy = (real_t)points[p1].y - (real_t)points[p2].y;
		real_t sqLen = std::max<real_t>(dx*dx + dy*dy, (real_t)0.0001);
		real_t fact = mParams.vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen));	// mul factor
		dx *= fact;
		dy *= fact;
		forces[p1].x += dx;
		forces[p1].y += dy;
		forces[p2].x -= dx;
		forces[p2].y -= dy;
	}

	void addCompulsiveForce(const std::vector<point_t> &points, index_t p1, index_t p2, length_t length, std::vector<point_t> &forces)
	{
		real_t dx = (real_t)points[p2].x - (real_t)points[p1].x;
		real_t dy = (real_t)points[p2].y - (real_t)points[p1].y;
		real_t sqLen = dx*dx + dy*dy;
		real_t fact = (real_t)std::sqrt(sqLen) * mParams.edgeCompulsion / (real_t)(length);
		dx *= fact;
		dy *= fact;
		forces[p1].x += dx;
		forces[p1].y += dy;
		forces[p2].x -= dx;
		forces[p2].y -= dy;
	}

    const std::vector<edge_t> &mEdges;		///< Reference to the graph edges.
	const std::vector<length_t> &mLengths;	///< Reference to the graph lengths of the edges.
	std::vector<point_t> mVelocities;			///< Point velocity vectors.
	std::vector<point_t> mForces;				///< Preallocated buffer for force vectors.
	ModelParameters<F> mParams;

	void updateVelocities(const std::vector<point_t> &forces)
	{
		real_t fact = mParams.timeQuantum / mParams.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
		for (std::size_t i = 0; i < mVelocities.size(); ++i) {
			mVelocities[i].x = (mVelocities[i].x + (real_t)forces[i].x * fact) * mParams.slowdown;
			mVelocities[i].y = (mVelocities[i].y + (real_t)forces[i].y * fact) * mParams.slowdown;
		}
	}


	void computeForces(std::vector<point_t> &points, std::vector<point_t> &forces)
	{
		forces.resize(points.size());

		for (std::size_t i = 0; i < forces.size(); ++i) {
			forces[i].x = forces[i].y = (real_t)0.0;
		}

		for (index_t i = 1; i < forces.size(); ++i) {
			for (index_t j = 0; j < i; ++j)
				addRepulsiveForce(points, i, j, forces);
		}

		for (std::size_t i = 0; i < mEdges.size(); ++i)
			addCompulsiveForce(points, mEdges[i].p1, mEdges[i].p2, mLengths[i], forces);
	}

public:
	SerialSimulator(index_t pointCount, const std::vector<edge_t> &edges,
		const std::vector<length_t> &lengths, const ModelParameters<real_t> &params)
		: mEdges(edges), mLengths(lengths), mParams(params)
	{
		mVelocities.resize(pointCount);
		mForces.resize(pointCount);
	}



	void updatePoints(std::vector<point_t> &points)
	{
		if (points.size() != mVelocities.size())
			throw (bpp::RuntimeError() << "Cannot compute next version of point positions."
				<< "Current model uses " << mVelocities.size() << " points, but the given buffer has " << points.size() << " points.");

		computeForces(points, mForces);
		updateVelocities(mForces);

		for (std::size_t i = 0; i < mVelocities.size(); ++i) {
			points[i].x += mVelocities[i].x * mParams.timeQuantum;
			points[i].y += mVelocities[i].y * mParams.timeQuantum;
		}
	}
};

#endif
