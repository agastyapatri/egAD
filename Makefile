main: main.c egad.c 
	clang -std=c17 -Wall -Wextra -O3 main.c egad.c -o main -lm 
