# Accelerated-Gradient-Boosting

Gradient Boosting model are suppose to provide stellar performance for supervised learning.Recently it has been proven that we can have faster convergence with Nestrov's momentum sequence for optimization.This is the [paper](https://arxiv.org/abs/1803.02042).

Author has shown the results in R. This is the python implementation of the same.

Following is the algorithm:-


\begin{algorithm}
\caption{Gradient Tree Boosting Algorithm}\label{alg:euclid}
\begin{algorithmic}[1]
\Procedure{GBT}{$x,y$}
\Initialize {$f_0(x) = arg min_{\gamma}\sum_{i=1}^N L(y_i,\gamma)$}
\For{\texttt{m = 1 to M:}}
        \For{\texttt{i = 1 to N: compute}}
        \State $r_{im}$ = $-\Big[\frac{\delta L(y_i,f(x_i))}{\delta f(x_i)}\Big]$
        
		\EndFor
        \State Fit a regression tree to the targets $r_{im}$ giving terminal regions $R_{jm}$, j = 1,2,...,$J_m$
        \For{\texttt{j = 1 to $J_m$: compute}}
        \State $\gamma_{jm}$= $arg min_{\gamma}$ $\sum_{x_i \epsilon R_{jm}}$ $L(y_i,f_{m-1}(x_i) + \gamma)$
        
		\EndFor
        
\Update {$f_m(x)$ = $f_{m-1}(x)$ + $\sum_{j=1}^{J_m}\gamma_{jm}I(x \epsilon R_{jm})$}
        
\EndFor
\State Output $\hat{f(x)}$ = $f_M(x)$
\EndProcedure
\end{algorithmic}
\end{algorithm}


