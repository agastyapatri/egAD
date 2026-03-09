#   egAD! 
(04/03/2026): 


A simple scalar valued autodifferentiation engine in C, written mostly for self edification.


The core of this library is the structure: 
```c 
typedef struct scalar {
	OPTYPE op;                              // operation which yielded the scalar 
	struct scalar* previous[NUM_PREVS];     // parent nodes in the graph. 
	double data;                            
	graph* tape;                            // computational graph in which a scalar instance forms a node.
	double grad;                            // derivative of the child node with respect to the last computation tracked by the graph.
} scalar;
```
Working in the background is: 

```c 
typedef struct graph {
	scalar** nodes;         // scalars which have been automatically added to the graph after initialization
	size_t num_nodes;       
	int* ref_count;         // the graph is only freed when there are no nodes which refer to it, more of a sanity check than anything
} graph;
```
This library also provides automatic operation tracking and mathematics on the `scalar` instances. To dynamically build the computational graph, it needs to be initialized before the nodes are added to it:

```c 
graph* compgraph = graph_init();
scalar* node1 = scalar_init(10.0, NONE, compgraph);
```
A couple of things to note: 

1.  `OPTYPE NONE` signifies that the scalar being created is a leaf node.  
2.  Any scalars that are created using the graph's leaf nodes are automatically attached to it.

While it is not strictly necessary to use `struct graph` - it is entirely possible to traverse the current node's adjacent nodes without it - it forms a helpful utility to see the true "forward pass" of the set of operations that have taken place with `graph_print()` after it has been sorted.


The actual gradient calculation is done by the `backward()`. `main.c` contains a simple example.
