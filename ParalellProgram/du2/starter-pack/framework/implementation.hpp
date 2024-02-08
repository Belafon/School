#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP


#include <tbb/tbb.h>
#include <iostream>

template<typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{

private:
	typedef typename POINT::coord_t coord_t;

	static coord_t distance(const POINT &point, const POINT &centroid)
	{
		std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
		std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
		return (coord_t)(dx*dx + dy*dy);
	}

	static std::size_t getNearestCluster(const POINT &point, const std::vector<POINT> &centroids)
	{
		coord_t minDist = distance(point, centroids[0]);
		std::size_t nearest = 0;
		for (std::size_t i = 1; i < centroids.size(); ++i) {
			coord_t dist = distance(point, centroids[i]);
			if (dist < minDist) {
				minDist = dist;
				nearest = i;
			}
		}

		return nearest;
	}

    // Define the reduction object to accumulate the partial sums of the cluster coordinates
    struct SumReducer {
        const std::vector<POINT>& centroids;
        const std::vector<POINT>& points;
        std::vector<ASGN>& assignments;

        SumReducer(const std::vector<POINT> &centroids_, const std::vector<POINT>& points_, std::vector<ASGN>& assignments_)
         : centroids(centroids_), points(points_), assignments(assignments_) {
            //std::cout << "SumReducer initialize..." << std::endl;

            partial_sums.resize(centroids.size());
            counts.resize(centroids.size());
        }

        SumReducer(SumReducer& other, tbb::split) 
        : centroids(other.centroids), points(other.points), 
            assignments(other.assignments) {
            //std::cout << "SumReducer new partition..." << std::endl;
            partial_sums.resize(centroids.size());
            counts.resize(centroids.size());
        }

        void operator()(const tbb::blocked_range<std::size_t>& range) {
            //std::cout << "SumReducer operation..." << std::endl;

            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                std::size_t nearest = getNearestCluster(points[i], centroids);
                assignments[i] = (ASGN)nearest;
                partial_sums[nearest].x += points[i].x;
                partial_sums[nearest].y += points[i].y;
                ////std::cout << "... " << i << " : " << counts[i] << std::endl;

                ++counts[nearest];
            }
        }

        void join(const SumReducer& other) {
            //std::cout << "SumReducer join... " << partial_sums.size() << " " << other.partial_sums.size() << std::endl;

            for (std::size_t i = 0; i < partial_sums.size(); ++i) {
                partial_sums[i].x += other.partial_sums[i].x;
                partial_sums[i].y += other.partial_sums[i].y;
                counts[i] += other.counts[i];
            }
        }
        /// @brief pocet bodu v danem clusteru
    	std::vector<std::size_t> counts;
        std::vector<POINT> partial_sums;
    };

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(std::size_t points, std::size_t k, std::size_t iters)
	{
        //std::cout << "initialize..." << std::endl;
        //counts.resize(k);
    }


	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
		std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
        //std::cout << "compute..." << std::endl;
		// Prepare for the first iteration
		centroids.resize(k);
		assignments.resize(points.size());
		for (std::size_t i = 0; i < k; ++i)
			centroids[i] = points[i];

		// Run the k-means refinements
		while (iters > 0) {
			--iters;
            //std::cout << "iteration: " << iters << " ..." << std::endl;
            
            SumReducer reducer(centroids, points, assignments);
            tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, points.size(), 4), reducer);
            //std::cout << "...parallel reduce done "<< counts.size() << std::endl;

			for (std::size_t i = 0; i < k; ++i) {
				if (reducer.counts[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.

				centroids[i].x = reducer.partial_sums[i].x / (std::int64_t)reducer.counts[i];
				centroids[i].y = reducer.partial_sums[i].y / (std::int64_t)reducer.counts[i];
			}

		}
		/* /for (std::size_t i = 0; i < k; ++i) {
            std::cout << i << " ... " << centroids[i].x << ", " << centroids[i].y << std::endl;
        } */
        /* for (size_t l = 0; l < k; l++)
        {
            for (size_t i = 0; i < assignments.size(); i++){
                if(assignments[i] == l)
                    std::cout << i << std::endl;
            }
            std::cout << std::endl;
        }
             */
	}
};


#endif
