//	A ad_value ad_valued autodifferentiation library
#ifndef EGAD_H
#define EGAD_H 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdbool.h> 
#define NUM_PREVS 2
#define NEXT 2
#define GRAPH_SIZE 256
#define PI 3.1415926545897932
#define MU 0 
#define SIGMA 0.08
#define GRAPH_EQUALITY(inp1, inp2) (inp1->tape == inp2->tape) 

typedef unsigned int uint;

typedef enum {
	NONE,
	ADD,
	SUB,
	MUL,
	DIV,
	POW,
	SIGMOID,
	TANH,
	RELU,
	SIN,
	COS,
	LOG, 
	EXP, 
} OPTYPE;

typedef struct ad_value {
	OPTYPE op;
	double data;
	double grad;
	int ref_count;
	struct ad_value* previous[NUM_PREVS]; 
} ad_value;


const char* get_optype_string(OPTYPE op);
ad_value* ad_value_random_gauss(double mu, double sigma);
ad_value* ad_value_alloc	(double data);
ad_value* ad_value_rand_normal(double mu, double sigma);
void ad_value_print		(ad_value* val);
void ad_value_free		(ad_value* val);

ad_value* ad_value_add	(ad_value* inp1, ad_value* inp2);
ad_value* ad_value_sub	(ad_value* inp1, ad_value* inp2);
ad_value* ad_value_mul	(ad_value* inp1, ad_value* inp2);
ad_value* ad_value_div	(ad_value* inp1, ad_value* inp2);
ad_value* ad_value_pow	(ad_value* inp1, ad_value* exponent);
ad_value* ad_value_sigmoid(ad_value* inp1);
ad_value* ad_value_tanh	(ad_value* inp1);
ad_value* ad_value_log	(ad_value* inp1);
ad_value* ad_value_exp	(ad_value* inp1);
ad_value* ad_value_sin	(ad_value* inp1);
ad_value* ad_value_cos	(ad_value* inp1);
ad_value* ad_value_relu	(ad_value* inp1);
bool ad_value_equality 	(ad_value* inp1, ad_value* inp2);
void ad_backward 	(ad_value* out);
double rand_normal(double mu, double sigma);
#endif
