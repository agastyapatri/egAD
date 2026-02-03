main: main.c autodiff.c 
	clang -std=c17 -Wall -Wextra -O3 main.c autodiff.c -o main -lm 
