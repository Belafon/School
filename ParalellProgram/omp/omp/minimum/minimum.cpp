#include "omp.h"

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstdint>



/**
 * Generate random numbers into an array.
 */
template<typename T>
void generate_data(std::vector<T> &data, std::size_t count = 0)
{
	if (count != 0) data.resize(count);

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<T> distribution(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

	for (auto && i : data) i = distribution(generator);
}


template<typename T>
T serial_min(const std::vector<T> &data)
{
	T res = data.front();
	for (auto i : data) res = std::min(res, i);
	return res;
}

template<typename T>
T para_min(const std::vector<T>& data)
{
	/*
        T res = data.front();
        for(auto i:data) res = std::min(res, i);
        return res;
    */
    int localMinimum = data.front();
    int containerSize = data.size();
    int min;
    #pragma omp parallel private(localMinimum) shared(containerSize) reduction(min:min)
    {
        #pragma omp for
        for (int i = 0; i < containerSize; ++i){
            localMinimum = std::min(localMinimum, i);
        }
        min = localMinimum;
    }

    return localMinimum;
}

int main(int argc, char *argv[])
{
	typedef std::int32_t number_t;

	std::cout << "Generating data ..." << std::endl;
	std::vector<number_t> data;
	generate_data(data, 2048 * 2048);


	double tstart, tend;

	std::cout << "Finding minimum serially ... ";
	std::cout.flush();
	tstart = omp_get_wtime();
	number_t serialmin = serial_min(data);
	tend = omp_get_wtime();
	std::cout << serialmin << " (" << (tend - tstart) << "s)" << std::endl;

	std::cout<<"Finding parallel minimum ... ";
	std::cout.flush();
	tstart = omp_get_wtime();
	number_t paramin = para_min(data);
	tend = omp_get_wtime();
	std::cout<<paramin<<" ("<<(tend-tstart)<<"s)"<<std::endl;

	return 0;
}

