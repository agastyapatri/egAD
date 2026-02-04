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
	graph_print(cgraph);
	printf("\n\n");
	backward(f);
	printf("%lf\n", f->grad);
	printf("%lf\n", e->grad);
	printf("%lf\n", d->grad);
	printf("%lf\n", c->grad);
	printf("%lf\n", b->grad);
	printf("%lf\n", a->grad);
	


	return 0;
}










