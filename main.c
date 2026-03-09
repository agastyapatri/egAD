#include <stdio.h>
#include <math.h>
#include <string.h>
#include "egad.h"





int main(){
	ad_value* a = ad_value_alloc(2.0);
	ad_value* b = ad_value_alloc(3.0);
	ad_value* c = ad_value_sin(a);
	ad_value* d = ad_value_cos(b);
	ad_value* e = ad_value_mul(c, d);
	ad_value* alpha = ad_value_alloc(5.0);
	ad_value* f = ad_value_sigmoid(e);
	ad_value* g = ad_value_relu(f);
	ad_value* h = ad_value_exp(g);
	ad_value* i = ad_value_pow(h, alpha);
	ad_value* j = ad_value_sigmoid(i);

	ad_backward(j);

	return 0;
}

