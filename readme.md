#   egAD! 
A simple scalar valued autodifferentiation engine written in C. 


**TODO** 
1.  Write the topological sorting algorithm for unsorted graphs  
2.  add more ops to `OPTYPE`
3.  write more tests for complicated math.
4.  look at the structure of `HIPS/autograd` to get inspiration
5.  look into how a graph can be implemented as an arena allocator, and the different nodes attached to it would just be pointers being assigned to it.


##  TOPOLOGICAL SORTING
Toposort finds a "Topological Order" - permutation of the nodes of the graph which corresponds to the order defined by all the edges of the graph. Every edge leads from node with a smaller vertex to a node with a larger one.

Topological order can be non unique( if there exist three nodes a, b, c for which there exist paths from a to b and a to c but not paths from b to c or c to b) 

Topological order only exists if the drected graph contains no cycles. Luckily in autodifferentiation, the nodes form a Directed Acyclic Graph. 

If there is a cycle in the graph between nodes a and b, a will need to have a smaller index than b (because an edge goes from a to b), and a will also need to have a larger index than b (because there is an edge going from b to a). 

**Every Directed Acyclic Graph contains at least one topological order**

*   A common situtation which necessitates toposort: 
    
There are n variables with unknown values. For some variables, we know that one of them is less than the other. You have to check whether these constraints are contradictory, and if not, output the variables in ascending order.

### The Algorithm
Toposort uses DFS. 
When astarting from some vertex v, DFS traverses along all edges outgoing from v. It stop at the edges for which the nodes have already been visited, and traverses along the rest of the edges and continues recursively at their nodes. 

Thus, by the time of the function call `dfs(v)` has finished, all vertices that are reachable from v have been either directly or indirectly been visited by the search. 

Let's append the vertex v to a list. Since all reachable vertices from v have been visited, they will have been appended to the list. If this is done for every vertex in the graph, with one or multiple DFS runs, For every edge
`v->u` in the graph, `u` will appear earlier in this list than v, because u is reachable from v. 




















