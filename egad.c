#include "egad.h"

const char* get_optype_string(OPTYPE op){
	switch (op) {
		case(ADD):
			return "add";
		case(SUB):
			return "sub";
		case(MUL):
			return "mul";
		case(POW):
			return "pow";
		case (SIGMOID):
			return "sigmoid";
		case (TANH):
			return "tanh";
		case (RELU):
			return "relu";
		case (NONE):
			return "none";
		case (SIN):
			return "sin";
		case (COS):
			return "cos";
	} 
	return NULL;
}

scalar* scalar_init(double data, OPTYPE operation, graph* tape){
	scalar* a = (scalar*)malloc(sizeof(scalar));
	a->data = data; 
	a->grad = 0;
	a->op = operation;
	a->previous[0] = NULL;
	a->previous[1] = NULL;
	a->tape = tape;
	if(tape){
		graph_push_back(a->tape, a);
		(*(a->tape->ref_count))++;
	}
	return a;
}

void scalar_free(scalar* val){
	if(val->tape){
		(*(val->tape->ref_count))--;
	}
	free(val);
}


void scalar_print(scalar* val){
	printf("scalar(%lf, %s)", val->data, get_optype_string(val->op));
}

scalar* scalar_add(scalar* inp1, scalar* inp2){
	scalar* out = scalar_init(inp1->data + inp2->data, ADD, inp1->tape);
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	return out;
}

scalar* scalar_sub(scalar* inp1, scalar* inp2){
	scalar* out = scalar_init(inp1->data - inp2->data, SUB, inp1->tape);
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	return out;
}
scalar* scalar_mul(scalar* inp1, scalar* inp2){
	scalar* out = scalar_init(inp1->data * inp2->data, MUL, inp1->tape);
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	return out;
}

scalar* scalar_pow(scalar* inp1, scalar* exponent){
	scalar* out = scalar_init(pow(inp1->data, exponent->data), POW, inp1->tape);
	out->previous[0] = inp1;
	out->previous[1] = exponent;

	return out;
}

scalar* scalar_sigmoid(scalar* inp1){
	double sigm = 1.0 / (1 + exp(-(inp1->data)));
	scalar* out = scalar_init(sigm, SIGMOID, inp1->tape);
	out->previous[0] = inp1;
	return out;
}

scalar* scalar_tanh(scalar* inp1){
	scalar* out = scalar_init(tanh(inp1->data), TANH, inp1->tape);
	out->previous[0] = inp1;
	return out;
}

scalar* scalar_sin(scalar* inp1){
	scalar* out = scalar_init(sin(inp1->data), SIN, inp1->tape);
	out->previous[0] = inp1;
	return out;
}

scalar* scalar_cos(scalar* inp1){
	scalar* out = scalar_init(cos(inp1->data), COS, inp1->tape);
	out->previous[0] = inp1;
	return out;
}


scalar* scalar_relu(scalar* inp1){
	scalar* out = scalar_init((inp1->data > 0) ? inp1->data : 0, RELU, inp1->tape);
	out->previous[0] = inp1;
	return out;
}

bool scalar_equality(scalar* inp1, scalar* inp2){
	return (inp1->data == inp2->data) && (inp1->grad == inp2->grad);
}


void grad(scalar* out){
	switch(out->op){
		case NONE: 
			break; 
		case ADD: 
			out->previous[0]->grad += out->grad; 
			out->previous[1]->grad += out->grad; 
			break; 
		case SUB: 
			out->previous[0]->grad += out->grad; 
			out->previous[1]->grad += -out->grad; 
			break; 
		case MUL: 
			out->previous[0]->grad += out->grad * out->previous[1]->data; 
			out->previous[1]->grad += out->grad * out->previous[0]->data; 
			break; 
		case POW:
			out->previous[0]->grad += out->grad * out->previous[1]->data * pow(out->previous[0]->data, out->previous[1]->data - 1) ;
			out->previous[1]->grad += out->grad * (out->data * log(out->previous[0]->data)) ;
			break;
		case SIN: 
			out->previous[0]->grad += out->grad * cos(out->previous[0]->data);
			break; 
		case COS: 
			out->previous[0]->grad += out->grad * -1 * sin(out->previous[0]->data);
			break; 
		case TANH: 
			out->previous[0]->grad += out->grad * (1 - pow(out->data, 2));
			break; 
		case SIGMOID: 
			out->previous[0]->grad += out->grad * out->data * (1 - out->data);
			break; 
		case RELU: 
			out->previous[0]->grad += out->grad * ((out->data > 0 ) ? 1 : 0);
			break; 
	}
}

graph* graph_init(){
	graph* g = (graph*)malloc(sizeof(graph));
	g->num_nodes = 0;
	g->nodes = (scalar**)calloc(GRAPH_SIZE, sizeof(scalar*));
	g->ref_count = (int*)calloc(1, sizeof(int));
	// (*(g->ref_count))++;
	return g;
}

void graph_push_back(graph* tape, scalar* val){
	tape->nodes[tape->num_nodes] = val;
	tape->num_nodes++;
	if((tape->num_nodes+1) % GRAPH_SIZE == 0){
		tape->nodes = realloc(tape->nodes, (tape->num_nodes+GRAPH_SIZE)*sizeof(scalar*));
	}
}

void graph_print(graph* tape){
	printf("Graph([\n");
	for(size_t i = 0; i < tape->num_nodes; i++){
		// if(i != tape->num_nodes-1)
		// 	printf("\t");
		printf("\t");
		scalar_print(tape->nodes[i]);
		printf("\n");
	}
	printf("])\n");
}

void graph_free(graph* tape){
	if(tape->ref_count == 0){
		for(size_t i = 0; i < tape->num_nodes; i++)
			free(tape->nodes[i]);
		free(tape->nodes);
		free(tape);
	}
}

void backward(scalar* out){
	out->grad = 1;
	scalar* temp = out;
	while(temp){
		grad(temp);
		temp = temp->previous[0];
	}
	free(temp);
}
