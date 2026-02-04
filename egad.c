#include "egad.h"
scalar* scalar_init(double data, OPTYPE operation){
	scalar* a = (scalar*)malloc(sizeof(scalar));
	a->data = data; 
	a->grad = 0;
	a->op = operation;
	a->previous[0] = NULL;
	a->previous[1] = NULL;
	return a;

}

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

void scalar_print(scalar* val){
	printf("scalar(%lf, %s)", val->data, get_optype_string(val->op));
}

scalar* scalar_add(scalar* inp1, scalar* inp2){
	scalar* out = scalar_init(inp1->data + inp2->data, ADD);
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	return out;
}

scalar* scalar_sub(scalar* inp1, scalar* inp2){
	scalar* out = scalar_init(inp1->data - inp2->data, SUB);
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	return out;
}
scalar* scalar_mul(scalar* inp1, scalar* inp2){
	scalar* out = scalar_init(inp1->data * inp2->data, MUL);
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	return out;
}

scalar* scalar_pow(scalar* inp1, scalar* exponent){
	scalar* out = scalar_init(pow(inp1->data, exponent->data), POW);
	out->previous[0] = inp1;
	out->previous[1] = exponent;

	return out;
}

scalar* scalar_sigmoid(scalar* inp1){
	scalar* out = scalar_init(1 / (1 + exp(-1*inp1->data)), POW);
	out->previous[0] = inp1;
	return out;
}

scalar* scalar_tanh(scalar* inp1){
	scalar* out = scalar_init(tanh(inp1->data), TANH);
	out->previous[0] = inp1;
	return out;
}

scalar* scalar_sin(scalar* inp1){
	scalar* out = scalar_init(sin(inp1->data), SIN);
	out->previous[0] = inp1;
	return out;
}

scalar* scalar_cos(scalar* inp1){
	scalar* out = scalar_init(cos(inp1->data), COS);
	out->previous[0] = inp1;
	return out;
}


scalar* scalar_relu(scalar* inp1){
	scalar* out = scalar_init((inp1->data > 0) ? inp1->data : 0, RELU);
	out->previous[0] = inp1;
	return out;
}

bool scalar_equality(scalar* inp1, scalar* inp2){
	return (inp1->data == inp2->data) && (inp1->grad == inp2->grad);
}
/*
 *	z = x ^ y 
 *	ln(z) = y * ln(x) 
 *	(1/z)*(dz / dy) = ln(x) 
 *	y->grad = dz / dy = z * ln(x)
 * */ 
void scalar_backward(scalar* out){
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
	for(size_t i = 0; i < tape->num_nodes; i++)
		free(tape->nodes[i]);
	free(tape->nodes);
	free(tape);
}

void graph_backward(graph* tape){
	tape->nodes[tape->num_nodes - 1]->grad = 1;
	scalar* out = tape->nodes[tape->num_nodes - 1];
	while(out){
		scalar_backward(out);
		out = out->previous[0];
	}
}
