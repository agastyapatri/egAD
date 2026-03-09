#   egAD! 
A simple scalar valued, reverse-mode autodiff engine in C.


The core of this library is the structure: 
```c 
typedef struct ad_value {
	OPTYPE op;                              // operation which yielded the ad_value 
	struct ad_value* previous[NUM_PREVS];     // parent nodes in the graph. 
	double data;                            
	double grad;                            // derivative of the child node with respect to the last computation tracked by the graph.
} ad_value;
```

A couple of things to note: 

1.  `OPTYPE NONE` signifies that the ad_value being created is a leaf node.  
2.  Any ad_values that are created using the graph's leaf nodes are automatically attached to it.


The actual gradient calculation is done by the `ad_backward()`. `main.c` contains a simple example.
