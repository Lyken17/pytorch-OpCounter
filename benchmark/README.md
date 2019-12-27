# MACs, FLOPs, what is the difference?

`FLOPs` is abbreviation of **floating operations** which includes mul / add / div ... etc. 

`MACs` stands for  **multiply–accumulate operation** that performs `a <- a + (b x c)`.

As shown in the text, one `MACs` has one `mul` and one `add`. That is why in many places `FLOPs` is nearly two times as `MACs`.

However, the application in real world is far more complex. Let's consider a matrix multiplication example. 
`A` is an matrix of dimension `mxn` and `B` is an vector of `nx1`. 

```python
for i in range(m):
    for j in range(n):
        C[i][j] += A[i][j] * B[j] # one mul-add
```     

It would be `mn` `MACs` and `2mn` `FLOPs`. But such implementation is slow and parallelization is necessary to run faster


  ```python
for i in range(m):
    parallelfor j in range(n):
        d[j] = A[i][j] * B[j] # one mul
    C[i][j] = sum(d) # n adds
```

Then the number of `MACs` is no longer `mn` . 


When comparing MACs /FLOPs, we want the number to be implementation-agnostic and as general as possible. Therefore in THOP, **we only consider the number of multiplications** and ignore all other operations. 

PS: The FLOPs is approximated by multiplying two.