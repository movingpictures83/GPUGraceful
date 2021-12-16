#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "time.h"
#include "GPUGracefulPlugin.h"
using namespace std;

void GPUGracefulPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 NUMNODES = atoi(parameters["numnodes"].c_str());
 NUMPERMS = atof(parameters["numperms"].c_str());
 std::string stopsfile = std::string(PluginManager::prefix())+"/"+parameters["stops"];
 std::ifstream infile2(stopsfile.c_str(), std::ios::in);
 int val;
 stops = (int*) malloc(NUMNODES*sizeof(int));
 int i;
 for (i = 0; i < NUMNODES; i++)
 {
    infile2 >> val; 
    stops[i] = val;  
 }
 children = (int*) malloc((NUMNODES-1)*sizeof(int));
 labels = (int*) malloc(NUMNODES*sizeof(int));
 edges = (int*) malloc((NUMPERMS*(NUMNODES-1))*sizeof(int));
 perms = (int*) malloc((NUMPERMS*NUMNODES)*sizeof(int));
 graceful_labels = (int*) malloc(NUMPERMS*sizeof(int));
}

void GPUGracefulPlugin::run() {
   found = 0;
   bool has_next = false;
   bool has_started = false;

    iter = 0;
    // generate both children and label array
    for(int i = 0; i < NUMNODES; i++)
    {
        labels[i] = i;
        if(i < NUMNODES - 1) children[i] = i+1;
    }
do{
    // create all permutations of given nodes
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
        {
            perms[i*NUMNODES+j] = labels[j];
            //edges[i*NUMNODES+j] = 0;
        }
        graceful_labels[i] = -1;
        has_next = next_permutation(labels, labels+NUMNODES);
if(!has_next) break;
    }
    if(!has_started)
    {
    	has_started = true;
	init(NUMNODES);
    }
    int *d_perms, *d_children, *d_graceful_labels, *d_stops, *d_edges;

    // define sizes for convenience
    const size_t perm_size = NUMNODES*NUMPERMS*sizeof(int);
    const size_t edge_size = (NUMNODES-1)*NUMPERMS*sizeof(int);
    const size_t child_size = (NUMNODES-1)*sizeof(int);
    const size_t stop_size = NUMNODES*sizeof(int);
    const size_t label_size = NUMPERMS*sizeof(int);

    // 768 cores available on my home computer
    // 1024 cores available on starship
    int numCores = (NUMNODES * NUMPERMS)/ 1024 + 1;
    int numThreads = 1024;

    // Allocate memory on GPU
    cudaMalloc(&d_perms, perm_size);
    cudaMalloc(&d_edges, edge_size);
    cudaMalloc(&d_children, child_size);
    cudaMalloc(&d_stops, stop_size);
    cudaMalloc(&d_graceful_labels, label_size);

    // Copy over necessary arrays to GPU
    cudaMemcpy(d_perms, perms, perm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_children, children, child_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stops, stops, stop_size, cudaMemcpyHostToDevice);

    // Calculate edge labelings for each permutation
    calculate_edges<<<numCores, numThreads>>>(d_perms,
            d_children,
            d_stops,
            d_edges,
            NUMNODES,
            NUMPERMS);

    // Don't need these for the next step, so just free the memory up.
    cudaFree(&d_perms);
    cudaFree(&d_stops);
    cudaFree(&d_children);

    // For debugging  purposes only
//    cudaMemcpy(edges, d_edges, edge_size, cudaMemcpyDeviceToHost);

    // Now check the gracefulness of the given edge labelings.
    check_gracefulness<<<numCores, numThreads>>>(d_edges, d_graceful_labels, NUMNODES, NUMPERMS);

    // Copy back the evaluated labelings
    cudaMemcpy(graceful_labels, d_graceful_labels, label_size, cudaMemcpyDeviceToHost);

    // Free up the rest of the memory
    cudaFree(&d_graceful_labels);
    cudaFree(&d_edges);
    for(int i = 0; i < NUMPERMS; i++)
    {
        if(graceful_labels[i] != -1)
	{
		for(int j = 0; j < NUMNODES; j++)
		cout << perms[graceful_labels[i] + j] << " ";
		cout << endl;
		    found=1;
		break;
	}
	
    }

iter++;
}while(has_next && found != 1);

}

void GPUGracefulPlugin::output(std::string file) {
    cout << "Found " << found << " graceful labelings." << endl;
    cout << "Took " << iter << " iterations" << endl;
    free(stops);
    free(children);
    free(labels);
    free(edges);
    free(perms);
    free(graceful_labels);
}


PluginProxy<GPUGracefulPlugin> GPUGracefulPluginProxy = PluginProxy<GPUGracefulPlugin>("GPUGraceful", PluginManager::getInstance());

