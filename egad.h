//	A scalar scalard autodifferentiation library
#ifndef EGAD_H
#define EGAD_H 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdbool.h> 
#include <math.h> 
#define PREVS 2
#define NEXT 2
#define GRAPH_SIZE 16
#define GRAPH_EQUALITY(inp1, inp2) (inp1->tape == inp2->tape) 

struct scalar; 
struct graph; 
typedef struct scalar scalar; 
typedef struct graph graph;

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

struct scalar {
	OPTYPE op;
	struct scalar* previous[PREVS];
	double data;
	double grad; //	grad holds the value of the derivative of the child node with respect to the parent node
	graph* tape;
} ;


const char* get_optype_string(OPTYPE op);
scalar* scalar_init(double data, OPTYPE operation, graph* tape);
void scalar_print(scalar* val);
void scalar_free(scalar* val);

scalar* scalar_add(scalar* inp1, scalar* inp2);
scalar* scalar_sub(scalar* inp1, scalar* inp2);
scalar* scalar_mul(scalar* inp1, scalar* inp2);
scalar* scalar_pow(scalar* inp1, scalar* exponent);
scalar* scalar_sigmoid(scalar* inp1);
scalar* scalar_tanh(scalar* inp1);
scalar* scalar_sin(scalar* inp1);
scalar* scalar_cos(scalar* inp1);
scalar* scalar_relu(scalar* inp1);
void grad(scalar* out);
bool scalar_equality(scalar* inp1, scalar* inp2);




struct graph {
	scalar** nodes;
	size_t num_nodes;
	int* ref_count;
};

graph* graph_init();
void graph_push_back(graph* tape,scalar* val);
void graph_print(graph* tape);
void graph_free(graph* tape);
void backward(scalar* out);



#endif /* ifndef AUTODIFF_H */
