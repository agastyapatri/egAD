#include <complex.h>
#include <stdio.h>
#include "egad.h"





int main(){
	/*
	 *	a = 2.0 
	 *	b = 3.0 
	 *	c = sin(a)
	 *	d = cos(b)
	 *	e = c + d
	 *	f = 1 / (1 + exp(sin(a)*cos(b)))
	 */ 
	graph* cgraph = graph_init();
	scalar* a = scalar_init(2.0, NONE, cgraph);
	scalar* b = scalar_init(3.0, NONE, cgraph);
	scalar* c = scalar_sin(a);
	scalar* d = scalar_cos(b);
	scalar* e = scalar_mul(c, d);
	scalar* f = scalar_sigmoid(e);
	scalar* g = scalar_relu(f);
	scalar* h = scalar_exp(g);
	scalar* i = scalar_mul(h, g);
	scalar* j = scalar_sigmoid(i);
	backward(j);

	// printf("%lf,%lf\n", j->data, j->grad);
	// printf("%lf,%lf\n", i->data, i->grad);
	// printf("%lf,%lf\n", h->data, h->grad);
	printf("%lf,%lf\n", g->data, g->grad);
	
	graph_print(cgraph);











	return 0;
}










