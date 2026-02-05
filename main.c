#include <stdio.h>
#include <string.h>
#include "egad.h"





int main(){
	graph* cgraph = graph_init();
	scalar* a = scalar_init(2.0, NONE, cgraph);
	scalar* b = scalar_init(3.0, NONE, cgraph);
	scalar* c = scalar_sin(a);
	scalar* d = scalar_cos(b);
	// scalar* e = scalar_mul(c, d);
	// scalar* alpha = scalar_init(5.0, NONE, cgraph);
	// scalar* f = scalar_sigmoid(e);
	// scalar* g = scalar_relu(f);
	// scalar* h = scalar_exp(g);
	// scalar* i = scalar_pow(h, alpha);
	// scalar* j = scalar_sigmoid(i);

	backward(c);

	graph_print(cgraph);
	printf("\n\n%lf\n", cos(2.0));





	return 0;
}

