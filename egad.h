//	A scalar scalard autodifferentiation library
#ifndef AUTODIFF_H
#define AUTODIFF_H 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#define PREVS 2
#define NEXT 2
#define GRAPH_SIZE 16


typedef enum {
	NONE,
	ADD,
	SUB,
	MUL,
	POW,
	SIGMOID,
	TANH,
	RELU,
	SIN,
	COS
} OPTYPE;

typedef struct scalar {
	OPTYPE op;
	struct scalar* previous[PREVS];
	double data;
	double grad; //	grad holds the value of the derivative of the child node with respect to the parent node
} scalar;

const char* get_optype_string(OPTYPE op);
scalar* scalar_init(double data, OPTYPE operation);
void scalar_print(scalar* val);

scalar* scalar_add(scalar* inp1, scalar* inp2);
scalar* scalar_sub(scalar* inp1, scalar* inp2);
scalar* scalar_mul(scalar* inp1, scalar* inp2);
scalar* scalar_pow(scalar* inp1, scalar* exponent);
scalar* scalar_sigmoid(scalar* inp1);
scalar* scalar_tanh(scalar* inp1);
scalar* scalar_sin(scalar* inp1);
scalar* scalar_cos(scalar* inp1);
scalar* scalar_relu(scalar* inp1);

void scalar_backward(scalar* out);





typedef struct graph {
	scalar** nodes;
	size_t num_nodes;
} graph;

graph* graph_init();
void graph_push_back(graph* tape,scalar* val);
void graph_print(graph* tape);
void graph_free(graph* tape);
void graph_backward(graph* tape);



#endif /* ifndef AUTODIFF_H */
