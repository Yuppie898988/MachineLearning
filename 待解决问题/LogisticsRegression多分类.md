# Logistics回归多分类

## 多项回归模型（Y取值集合为$\{0,1,2,\cdots,K\}$）

$$
\begin{aligned}
&P(Y=k|x)=\frac{e^{w_k \cdot x}}{1+\displaystyle \sum_{k=0}^{K-1}e^{w_k \cdot x}},\quad k=0,1,2,K-1\\
&P(Y=K|x)=\frac{1}{1+\displaystyle \sum_{k=0}^{K-1}e^{w_k \cdot x}}
\end{aligned}
$$

注：$x\in\mathcal R^{n+1},w_k\in\mathcal R^{n+1}$。其中$x$最后一项为1，$w_k$最后一项为$b$

以Y取值集合为$\{0,1,2\}$为例
$$
\begin{aligned}
P(Y=0|x)=\frac{e^{w_0 \cdot x}}{1+\displaystyle \sum_{k=0}^{1}e^{w_k \cdot x}}\\
P(Y=1|x)=\frac{e^{w_1 \cdot x}}{1+\displaystyle \sum_{k=0}^{1}e^{w_k \cdot x}}\\
P(Y=2|x)=\frac{1}{1+\displaystyle \sum_{k=0}^{1}e^{w_k \cdot x}}
\end{aligned}
$$


## 极大似然函数

### 假设

- $$
  P(Y=k|x)=\pi_k(x)
  $$

- 样本点为$x_i$，样本数为$N$ 

- Y取值集合为$\{0,1,2,\cdots,K\}$

  
### 似然函数

$$
\prod_{i=0}^N \prod_{k=0}^{K}[\pi_k(x_i)]^{I(y_i=k)}
$$

以Y取值集合为$\{0,1,2\}$为例
$$
\prod_{i=0}^N [\pi_0(x_i)]^{I(y_i=0)}[\pi_1(x_i)]^{I(y_i=1)}[\pi_2(x_i)]^{I(y_i=2)}
$$

### 对数似然函数

$$
\begin{aligned}
	L(w)&=\sum_{i=0}^N \sum_{k=0}^K I(y_i=k)ln[\pi_k(x_i)]\\
	&=\sum_{i=0}^N \{[\sum_{k=0}^{K-1}I(y_i=k)w_k \cdot x_i]-\sum_{k=0}^K I(y_i=k)ln(1+\sum_{k=0}^{K-1} e^{w_k \cdot x})\}\\
	&=\sum_{i=0}^N \{[\sum_{k=0}^{K-1}I(y_i=k)w_k \cdot x_i]-ln(1+\sum_{k=0}^{K-1} e^{w_k \cdot x_i})\}\\
	&=\sum_{i=0}^N [\sum_{k=0}^{K-1}I(y_i=k)w_k \cdot x_i]-\sum_{i=0}^Nln(1+\sum_{k=0}^{K-1} e^{w_k \cdot x_i})
\end{aligned}
$$

以Y取值集合为$\{0,1,2\}$为例
$$
\begin{aligned}
L(w)&=\sum_{i=0}^N [I(y_i=0)w_0 \cdot x_i + I(y_i=1)w_1 \cdot x_i]-\sum_{i=0}^Nln(1+e^{w_0 \cdot x_i}+e^{w_1 \cdot x_i})\\
&=\sum_{i=0}^N I(y_i=0)w_0 \cdot x_i+\sum_{i=0}^N I(y_i=1)w_1 \cdot x_i - \sum_{i=0}^Nln(1+e^{w_0 \cdot x_i}+e^{w_1 \cdot x_i})\\
\end{aligned}
$$


其中$\displaystyle \sum_{i=0}^N I(y_i=0)w_0 \cdot x_i$表示：**$y_i$为0的样本点$x_i$与$w_0$内积的和**

### 偏导

$$
\begin{aligned}
\frac{\partial{L}}{\partial{w_k}}&=\sum_{i=0}^N [\sum_{k=0}^{K-1}I(y_i=k)x_i] - \sum_{i=0}^N \frac{x_ie^{w_k \cdot x_i}}{1+\sum_{k=0}^{K-1} e^{w_k \cdot x_i}}
\end{aligned}
$$

其中$\displaystyle \sum_{i=0}^N [\sum_{k=0}^{K-1}I(y_i=k)x_i]$表示：**$y_i$为$k$的样本点$x_i$的和，仍为向量**

