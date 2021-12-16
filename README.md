# GPUGraceful
# Language: CUDA
# Input: TXT
# Output: SCREEN
# Tested with: PluMA 1.0, CUDA 10

Perform a graceful labeling of nodes in a graph, if one is available

Original authors: Jake Lopez, Charlotte Farolan, John Perez

The plugin accepts as input a tab-delimited file of keyword-value pairs:
numnodes: Number of nodes in the graph
numperms: Number of different permutations to try
stops: File containing child positions (1 per row, for each node)

Output is to the screen, you can specify 'none'
