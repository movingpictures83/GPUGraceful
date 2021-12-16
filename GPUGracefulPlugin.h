#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUGracefulPlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
		int NUMNODES;
		int NUMPERMS;
		int* stops;
                std::map<std::string, std::string> parameters;
		int* children;
		int* labels;
		int* edges;
		int* perms;
		int* graceful_labels;
		int found;
		float iter;
};

int factorial(int n)
{
  return (n == 0 || n == 1) ? 1 : factorial(n-1) * n;
}

__global__ void calculate_edges(int *perms, int *children, int *stops, int *edges, int NUMNODES, int NUMPERMS)
{
    /*
       Since the permutation array is a flattened 2D array which was NUMNODES wide and NUMPERMS long, then we
       must start at the begining of every row which would be offset by NUMNODES.
   */
    int element = (blockIdx.x * blockDim.x + threadIdx.x) * NUMNODES; 
    int total = NUMNODES * NUMPERMS; // make sure we do not exceed the size of the permutation array
    int edge_counter = 0; // keep track of where in the edge array we are putting the next edge label
    int last_index = 0; // keep track of the last index from the stop array.
    int edge_start = (blockIdx.x * blockDim.x + threadIdx.x) * (NUMNODES-1); // calculate where in the edge array we should begin placing labels
    if(element < total)
    {
        // Only go thorugh each NUMNODE group of labels
        for(int i = element; i < element + NUMNODES; i++)
        {
            // check for sentinel value of -1
            if(stops[i % NUMNODES] != -1)
            {
                // If this is our first time we start at 0, otherwise we continue from the the last index
                for(int j = (last_index == 0) ? 0 : last_index+1; j<=stops[i % NUMNODES]; j++)
                    // place the absolute difference of each end point into the edge array
                    edges[edge_start + edge_counter++] = abs(perms[i] - perms[children[j] + element]);
                last_index = stops[i % NUMNODES];
            }
        } }
}

__global__ void check_gracefulness(int *edges, int *graceful_labels, int NUMNODES, int NUMPERMS)
{
    /*
       Go through edge array and check for any duplicates. If there are duplicates found, exit the loop and mark this label
       as being nongraceful , which is designated by a -1 in the label array. If no duplicates are found, the labeling is graceful and
       the index of the permutation is stored.
   */
    int element = (blockIdx.x * blockDim.x + threadIdx.x) * (NUMNODES-1); 
    int total = NUMNODES * NUMPERMS;
    bool graceful = true;
    if(element < total)
    {
        for(int i = element; i < element + NUMNODES-1; i++)
        {
            int current = edges[i];
            for(int j = i + 1; j < element + NUMNODES-1; j++)
            {
                if(current == edges[j])
                {
                    graceful = false;
                    break;
                }
            }
            if(!graceful) break;
        }
        if(graceful)
            graceful_labels[element / (NUMNODES-1)] = element/(NUMNODES-1)*NUMNODES;
        if(!graceful)
            graceful_labels[element / (NUMNODES-1)] = -1;
    }
}
