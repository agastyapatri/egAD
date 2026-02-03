#include <stdio.h>
#include "autodiff.h"
int main(){
	/*
	 *	a = 10
	 *	b = 20 
	 *	c = a  + b 
	 *	d = c^3
	 *
	 *
	 *	c.grad = dd/dc = 3*c^2 
	 *	a.grad = dd / da = dd/dc * dc/da = c.grad * dc / da 
	 *	b.grad = dd / db = dd/dc * dc/db = c.grad * dc / db
	 */ 
	scalar* a = scalar_init(2.0, NONE);
	scalar* b = scalar_init(3.0, NONE);
	scalar* c = scalar_sin(a);
	scalar* d = scalar_cos(b);
	scalar* e = scalar_add(c, d);
	graph* compgraph = graph_init();
	graph_push_back(compgraph, a);
	graph_push_back(compgraph, b);
	graph_push_back(compgraph, c);
	graph_push_back(compgraph, d);
	graph_push_back(compgraph, e);
	graph_backward(compgraph);





	return 0;
}










	// graph_push_back(compgraph, a);
	// graph_push_back(compgraph, b);
	// graph_push_back(compgraph, c);
	// graph_print(compgraph);
	// graph_free(compgraph);
	// compgraph = NULL;
