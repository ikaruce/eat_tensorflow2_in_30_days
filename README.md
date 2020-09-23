# TensorFlow2 30 일 안에 해치우기 ?🔥🔥

Click here for [Chinese Version（中文版）](#30天吃掉那只-tensorflow2)

**《30天吃掉那只TensorFlow2》**
* 🚀 github项目地址: https://github.com/lyhue1991/eat_tensorflow2_in_30_days
* 🐳 和鲸专栏地址: https://www.kesci.com/home/column/5d8ef3c3037db3002d3aa3a0 【代码可直接fork后云端运行，无需配置环境】

**《20天吃掉那只Pytorch》**
* 🚀 github项目地址: https://github.com/lyhue1991/eat_pytorch_in_20_days
* 🐳 和鲸专栏地址: https://www.kesci.com/home/column/5f2ac5d8af3980002cb1bc08 【代码可直接fork后云端运行，无需配置环境】


### 1. TensorFlow2 🍎 or Pytorch🔥

결론: 

**엔지니어는 TensorFlow2를 먼저 배우세요.**

**학생이나 연구자라면, 파이토치를 먼저 선택하세요.**

**시간의 여유가 있다면, 둘 다 배우는 것이 가장 좋습니다.**


이유:

* 1. **업계에서는 모델 구현이 가장 중요합니다. 중국의 주요 인터넷 기업에서 tensorflow 모델(pytorch가 아님)을 독점적으로 사용하는게 현재 상황입니다.** 더불어, 업계에서는 많이 쓰일 수 있는 모델을 선호합니다. 대부분의 경우, 조금만 수정해서 사용할 수 있는 모델을 사용합니다. 


* 2. **수많은 새로운 모델을 테스트 해야하기 때문에 빠르게 반복해서 개발하고 발표하는 것은 연구자에게 매우 중요합니다. pytorch는 tensorflow와 비교하여 접근하고 디버깅하는데 장점을 가지고 있습니다.** Pytorch는 최신의 결과가 많이 있고, 2019년 이후에 대학에서 가장 많이 사용되고 있습니다. 


* 3. 종합해보면, Tensorflow와 Pytorch는 현시대의 프로그래밍에서는 매우 유사합니다. 그래서 하나를 배우면, 다른 하나를 배우는데 도움이 됩니다. 둘 다를 배우는 것은 더 많은 공개된 모델을 다룰 수 있게 되고, 둘 사이를 전환하는데 도움이 됩니다. 

```python

```

### 2. Keras🍏 and tf.keras 🍎

결론: 

**Keras 는 2.3.0부터 개발되지 않을 것입니다. 그러니 tf.keras를 사용하세요.**


Keras는 딥러닝을 위한 고수준의 API입니다. Keras는 사용자가 좀 더 직관적으로 딥러닝 네트워크를 정의하고 학습할 수 있게 합니다. 

pip로 설치한 Keras 라이브러리는 tensorflow, theano, CNTK 등등에서 사용할 수 있는 API 입니다. 

tf.keras는 Tensorflow만을 위한 고 수준의 API이고, tensorflow의 저수준 API를 기반합니다. 

tf.keras에서 대부분의 함수는 Keras와 동일합니다.(그래서 다른 종류의 백엔드와 호환됩니다) tf.keras는 Keras 보다 Tensorflow에 더 충실합니다. 

keras는 구글에 인수되면서 2.3.0 이후로 업데이트 하지 않을 것입니다. 그러므로 지금부터는 pip로 Keras를 설치하는 대신 tf.keras를 사용해야 합니다. 

```python

```

### 3. 이 책을 읽기 전에 알아야 할 것이 있나요 📖?

**머신 러닝 혹은 딥러닝의 기초 지식과 Keras나 Tensorflow로 모델링한 경험을 가지고 있으면 좋습니다.**

**만약 머신 러닝 혹은 딥러닝에 대한 경험이 전혀 없다면, ["케라스 창시자에게 배우는 딥러닝"](https://book.naver.com/bookdb/book_detail.nhn?bid=14069088) 을 같이 읽기를 권장합니다.**


["케라스 창시자에게 배우는 딥러닝"](https://book.naver.com/bookdb/book_detail.nhn?bid=14069088) 은 Keras의 창시자 François Chollet가 썼습니다. 이 책은 머신러닝에 경험이 없는 독자들을 대상으로 Keras에 기반한 책입니다. 

"케라스 창시자에게 배우는 딥러닝" 은 이해하기 쉽고 다양한 예제들을 보여줍니다. **이 책에는 직관적 인 것을 딥 러닝으로 육성하는 데 초점을 맞추고 있으므로이 책에는 수학 방정식이 없습니다.**


```python

```

### 4. 이 책의 스타일 🍉 


**이 책은 사람 친화적인 참고서 입니다. 저자의 낮은 목표는 어려워서 포기 하지 않게 하는 것이고, 독자가 생각하게 하지 않는다가 높은 목표입니다.**

이 책은 기본적으로 Tensorflow 공식 문서를 따릅니다. 

하지만, 저자가 예제를 최적화하여 다시 만들었습니다. 

공식 문서가 예제와 가이드가 혼란스럽게 배열 되어 있어 어렵습니다. 그것과 달리 이 책은 독자들이 Tensorflow의 아키텍처에 따라 순차적으로 익힐 수 있도록 배열 되어있습니다. 명확한 경로와 예제에 쉽게 접근함으로 써 Tensorflow를 학습을 점진적으로 만듭니다. 

공식 문서의 자세한 설명된 코드와 달리 저자는 읽기 및 구현을 쉽게하기 위해 예제의 길이를 최소화하려고합니다. 또한 대부분의 코드 셀을 프로젝트에서 즉시 사용할 수 있습니다

**공식 문서에서 9시간이 걸린 난이도였다면 이 책으로 배운다면 3시간으로 단축할 수 있습니다.**

난이도의 차이는 아래 그림과 같습니다.

![](./data/30天吃掉那个TF2.0_en.jpg)


```python

```

### 5. 학습 방법  ⏰

**(1) 학습 계획**

저자는 이 책을 여유시간을 이용하도록 했습니다. 대부분의 독자는 모든 내용을 30일이면 완료할 수 있습니다. 

매일 30분에서 2시간 정도 필요합니다. 

Tensorflow2로 머신러닝을 구현할 때, 참조할 라이브러리로 사용할 수도 있습니다. 

**아래 파란 글씨를 클릭하면 해당 단원으로 이동합니다.**


|Date |Contents                                                       | Difficulties   | Est. Time | Update Status|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp;|[**Chapter 1: Modeling Procedure of TensorFlow**](./english/Chapter1.md)    |⭐️   |   0hour   |✅    |
|Day 1 |  [1-1 Example: Modeling Procedure for Structured Data](./english/Chapter1-1.md)    | ⭐️⭐️⭐️ |   1hour    |✅    |
|Day 2 |[1-2 Example: Modeling Procedure for Images](./english/Chapter1-2.md)    | ⭐️⭐️⭐️⭐️  |   2hours    |✅    |
|Day 3 |  [1-3 Example: Modeling Procedure for Texts](./english/Chapter1-3.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hours    |✅    |
|Day 4 |  [1-4 Example: Modeling Procedure for Temporal Sequences](./english/Chapter1-4.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hours    |✅    |
|&nbsp;    |[**Chapter 2: Key Concepts of TensorFlow**](./english/Chapter2.md)  | ⭐️  |  0hour |✅  |
|Day 5 |  [2-1 Data Structure of Tensor](./english/Chapter2-1.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅    |
|Day 6 |  [2-2 Three Types of Graph](./english/Chapter2-2.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hours    |✅    |
|Day 7 |  [2-3 Automatic Differentiate](./english/Chapter2-3.md)  | ⭐️⭐️⭐️   |   1hour    |✅    |
|&nbsp; |[**Chapter 3: Hierarchy of TensorFlow**](./english/Chapter3.md) |   ⭐️  |  0hour   |✅  |
|Day 8 |  [3-1 Low-level API: Demonstration](./english/Chapter3-1.md)   | ⭐️⭐️⭐️⭐️ |   1hour    |✅   |
|Day 9 |  [3-2 Mid-level API: Demonstration](./english/Chapter3-2.md)   | ⭐️⭐️⭐️   |   1hour    |✅  |
|Day 10 |  [3-3 High-level API: Demonstration](./english/Chapter3-3.md)  | ⭐️⭐️⭐️   |   1hour    |✅  |
|&nbsp; |[**Chapter 4: Low-level API in TensorFlow**](./english/Chapter4.md) |⭐️    | 0hour|✅  |
|Day 11|  [4-1 Structural Operations of the Tensor](./english/Chapter4-1.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hours    |✅   |
|Day 12|  [4-2 Mathematical Operations of the Tensor](./english/Chapter4-2.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|Day 13|  [4-3 Rules of Using the AutoGraph](./english/Chapter4-3.md)| ⭐️⭐️⭐️   |   0.5hour    | ✅  |
|Day 14|  [4-4 Mechanisms of the AutoGraph](./english/Chapter4-4.md)    | ⭐️⭐️⭐️⭐️⭐️   |   2hours    |✅  |
|Day 15|  [4-5 AutoGraph and tf.Module](./english/Chapter4-5.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|&nbsp; |[**Chapter 5: Mid-level API in TensorFlow**](./english/Chapter5.md) |  ⭐️  | 0hour|✅ |
|Day 16|  [5-1 Dataset](./english/Chapter5-1.md)   | ⭐️⭐️⭐️⭐️⭐️   |   2hours    |✅  |
|Day 17|  [5-2 feature_column](./english/Chapter5-2.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|Day 18|  [5-3 activation](./english/Chapter5-3.md)    | ⭐️⭐️⭐️   |   0.5hour    |✅   |
|Day 19|  [5-4 layers](./english/Chapter5-4.md)  | ⭐️⭐️⭐️   |   1hour    |✅  |
|Day 20|  [5-5 losses](./english/Chapter5-5.md)    | ⭐️⭐️⭐️   |   1hour    |✅  |
|Day 21|  [5-6 metrics](./english/Chapter5-6.md)    | ⭐️⭐️⭐️   |   1hour    |✅   |
|Day 22|  [5-7 optimizers](./english/Chapter5-7.md)    | ⭐️⭐️⭐️   |   0.5hour    |✅   |
|Day 23|  [5-8 callbacks](./english/Chapter5-8.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|&nbsp; |[**Chapter 6: High-level API in TensorFlow**](./english/Chapter6.md)|    ⭐️ | 0hour|✅  |
|Day 24|  [6-1 Three Ways of Modeling](./english/Chapter6-1.md)   | ⭐️⭐️⭐️   |   1hour    |✅ |
|Day 25|  [6-2 Three Ways of Training](./english/Chapter6-2.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|Day 26|  [6-3 Model Training Using Single GPU](./english/Chapter6-3.md)    | ⭐️⭐️   |   0.5hour    |✅   |
|Day 27|  [6-4 Model Training Using Multiple GPUs](./english/Chapter6-4.md)    | ⭐️⭐️   |   0.5hour    |✅  |
|Day 28|  [6-5 Model Training Using TPU](./english/Chapter6-5.md)   | ⭐️⭐️   |   0.5hour    |✅  |
|Day 29| [6-6 Model Deploying Using tensorflow-serving](./english/Chapter6-6.md) | ⭐️⭐️⭐️⭐️| 1hour |✅   |
|Day 30| [6-7 Call Tensorflow Model Using spark-scala](./english/Chapter6-7.md) | ⭐️⭐️⭐️⭐️⭐️|2hours|✅  |
|&nbsp;| [Epilogue: A Story Between a Foodie and Cuisine](./english/Epilogue.md) | ⭐️|0hour|✅  |

```python

```

**(2) 학습을 위한 개발환경**


모든 소스 코드는 jupiter에서 테스트 되었습니다. 리파지토리를 클론한 다음 jupiter에서 실행해가며 배우기를 권장합니다. 

독자가 마크다운을 jupyter에서 바로 열 수 있도록, 마크다운을 ipynb로 변환해주는 jupytext를 설치하기를 권장합니다. 

```python
#마크다운 파일을 ipynb로 변환해주는 jupytext를 설치하기를 권장합니다. 
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U jupytext
    
#이 책의 코드를 실행해서 테스트 해볼 수 있도록 Tensorflow 최신버전을 설치하기를 권장 합니다. 
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -U tensorflow
```

```python
import tensorflow as tf

#Note: all the codes are tested under TensorFlow 2.1
tf.print("tensorflow version:",tf.__version__)

a = tf.constant("hello")
b = tf.constant("tensorflow2")
c = tf.strings.join([a,b]," ")
tf.print(c)
```

```
tensorflow version: 2.1.0
hello tensorflow2
```

```python

```

### 6. 저자에게 문의하기 🎈🎈


**책이 도움이 되었고, 저자를 응원한다면, 이 리파지토리에 스타를 주고 친구들에게 알려주세요.😊** 

저자와 책 내용에 대해 이야기 하고 싶다면 위챗 계정 "Python与算法之美" (Elegance of Python and Algorithms) 에 메시지를 남겨주세요. 

![image.png](./data/Python与算法之美logo.jpg)

```python

```
# 30天吃掉那只 TensorFlow2

📚 gitbook电子书地址： https://lyhue1991.github.io/eat_tensorflow2_in_30_days

🚀 github项目地址：https://github.com/lyhue1991/eat_tensorflow2_in_30_days

🐳 kesci专栏地址：https://www.kesci.com/home/column/5d8ef3c3037db3002d3aa3a0


### 一，TensorFlow2 🍎 or Pytorch🔥

先说结论:

**如果是工程师，应该优先选TensorFlow2.**

**如果是学生或者研究人员，应该优先选择Pytorch.**

**如果时间足够，最好TensorFlow2和Pytorch都要学习掌握。**


理由如下：

* 1，**在工业界最重要的是模型落地，目前国内的大部分互联网企业只支持TensorFlow模型的在线部署，不支持Pytorch。** 并且工业界更加注重的是模型的高可用性，许多时候使用的都是成熟的模型架构，调试需求并不大。


* 2，**研究人员最重要的是快速迭代发表文章，需要尝试一些较新的模型架构。而Pytorch在易用性上相比TensorFlow2有一些优势，更加方便调试。** 并且在2019年以来在学术界占领了大半壁江山，能够找到的相应最新研究成果更多。


* 3，TensorFlow2和Pytorch实际上整体风格已经非常相似了，学会了其中一个，学习另外一个将比较容易。两种框架都掌握的话，能够参考的开源模型案例更多，并且可以方便地在两种框架之间切换。

```python

```

### 二，Keras🍏 and  tf.keras 🍎

先说结论：

**Keras库在2.3.0版本后将不再更新，用户应该使用tf.keras。**


Keras可以看成是一种深度学习框架的高阶接口规范，它帮助用户以更简洁的形式定义和训练深度学习网络。

使用pip安装的Keras库同时在tensorflow,theano,CNTK等后端基础上进行了这种高阶接口规范的实现。

而tf.keras是在TensorFlow中以TensorFlow低阶API为基础实现的这种高阶接口，它是Tensorflow的一个子模块。

tf.keras绝大部分功能和兼容多种后端的Keras库用法完全一样，但并非全部，它和TensorFlow之间的结合更为紧密。

随着谷歌对Keras的收购，Keras库2.3.0版本后也将不再进行更新，用户应当使用tf.keras而不是使用pip安装的Keras.

```python

```

### 三，本书📖面向读者 👼


**本书假定读者有一定的机器学习和深度学习基础，使用过Keras或者Tensorflow1.0或者Pytorch搭建训练过模型。**

**对于没有任何机器学习和深度学习基础的同学，建议在学习本书时同步参考学习《Python深度学习》一书。**

《Python深度学习》这本书是Keras之父Francois Chollet所著，该书假定读者无任何机器学习知识，以Keras为工具，

使用丰富的范例示范深度学习的最佳实践，该书通俗易懂，**全书没有一个数学公式，注重培养读者的深度学习直觉。**。

```python

```

### 四，本书写作风格 🍉


**本书是一本对人类用户极其友善的TensorFlow2.0入门工具书，不刻意恶心读者是本书的底限要求，Don't let me think是本书的最高追求。**

本书主要是在参考TensorFlow官方文档和函数doc文档基础上整理写成的。

但本书在篇章结构和范例选取上做了大量的优化。

不同于官方文档混乱的篇章结构，既有教程又有指南，缺少整体的编排逻辑。

本书按照内容难易程度、读者检索习惯和TensorFlow自身的层次结构设计内容，循序渐进，层次清晰，方便按照功能查找相应范例。

不同于官方文档冗长的范例代码，本书在范例设计上尽可能简约化和结构化，增强范例易读性和通用性，大部分代码片段在实践中可即取即用。

**如果说通过学习TensorFlow官方文档掌握TensorFlow2.0的难度大概是9的话，那么通过学习本书掌握TensorFlow2.0的难度应该大概是3.**

谨以下图对比一下TensorFlow官方教程与本教程的差异。

![](./data/30天吃掉那个TF2.0.jpg)


```python

```

### 五，本书学习方案 ⏰

**1，学习计划**

本书是作者利用工作之余和疫情放假期间大概2个月写成的，大部分读者应该在30天可以完全学会。

预计每天花费的学习时间在30分钟到2个小时之间。

当然，本书也非常适合作为TensorFlow的工具手册在工程落地时作为范例库参考。

**点击学习内容蓝色标题即可进入该章节。**


|日期 | 学习内容                                                       | 内容难度   | 预计学习时间 | 更新状态|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp;|[**一、TensorFlow的建模流程**](./一、TensorFlow的建模流程.md)    |⭐️   |   0hour   |✅    |
|day1 |  [1-1,结构化数据建模流程范例](./1-1,结构化数据建模流程范例.md)    | ⭐️⭐️⭐️ |   1hour    |✅    |
|day2 |[1-2,图片数据建模流程范例](./1-2,图片数据建模流程范例.md)    | ⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|day3 |  [1-3,文本数据建模流程范例](./1-3,文本数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|day4 |  [1-4,时间序列数据建模流程范例](./1-4,时间序列数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|&nbsp;    |[**二、TensorFlow的核心概念**](./二、TensorFlow的核心概念.md)  | ⭐️  |  0hour |✅  |
|day5 |  [2-1,张量数据结构](./2-1,张量数据结构.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅    |
|day6 |  [2-2,三种计算图](./2-2,三种计算图.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅    |
|day7 |  [2-3,自动微分机制](./2-3,自动微分机制.md)  | ⭐️⭐️⭐️   |   1hour    |✅    |
|&nbsp; |[**三、TensorFlow的层次结构**](./三、TensorFlow的层次结构.md) |   ⭐️  |  0hour   |✅  |
|day8 |  [3-1,低阶API示范](./3-1,低阶API示范.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|day9 |  [3-2,中阶API示范](./3-2,中阶API示范.md)   | ⭐️⭐️⭐️   |  1hour    |✅  |
|day10 |  [3-3,高阶API示范](./3-3,高阶API示范.md)  | ⭐️⭐️⭐️  |   1hour    |✅  |
|&nbsp; |[**四、TensorFlow的低阶API**](./四、TensorFlow的低阶API.md) |⭐️    | 0hour|✅  |
|day11|  [4-1,张量的结构操作](./4-1,张量的结构操作.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅   |
|day12|  [4-2,张量的数学运算](./4-2,张量的数学运算.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|day13|  [4-3,AutoGraph的使用规范](./4-3,AutoGraph的使用规范.md)| ⭐️⭐️⭐️   |   0.5hour    |✅  |
|day14|  [4-4,AutoGraph的机制原理](./4-4,AutoGraph的机制原理.md)    | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅  |
|day15|  [4-5,AutoGraph和tf.Module](./4-5,AutoGraph和tf.Module.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|&nbsp; |[**五、TensorFlow的中阶API**](./五、TensorFlow的中阶API.md) |  ⭐️  | 0hour|✅ |
|day16|  [5-1,数据管道Dataset](./5-1,数据管道Dataset.md)   | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅  |
|day17|  [5-2,特征列feature_column](./5-2,特征列feature_column.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|day18|  [5-3,激活函数activation](./5-3,激活函数activation.md)    | ⭐️⭐️⭐️   |   0.5hour    |✅   |
|day19|  [5-4,模型层layers](./5-4,模型层layers.md)  | ⭐️⭐️⭐️   |   1hour    |✅  |
|day20|  [5-5,损失函数losses](./5-5,损失函数losses.md)    | ⭐️⭐️⭐️   |   1hour    |✅  |
|day21|  [5-6,评估指标metrics](./5-6,评估指标metrics.md)    | ⭐️⭐️⭐️   |   1hour    |✅   |
|day22|  [5-7,优化器optimizers](./5-7,优化器optimizers.md)    | ⭐️⭐️⭐️   |   0.5hour    |✅   |
|day23|  [5-8,回调函数callbacks](./5-8,回调函数callbacks.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|&nbsp; |[**六、TensorFlow的高阶API**](./六、TensorFlow的高阶API.md)|    ⭐️ | 0hour|✅  |
|day24|  [6-1,构建模型的3种方法](./6-1,构建模型的3种方法.md)   | ⭐️⭐️⭐️   |   1hour    |✅ |
|day25|  [6-2,训练模型的3种方法](./6-2,训练模型的3种方法.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|day26|  [6-3,使用单GPU训练模型](./6-3,使用单GPU训练模型.md)    | ⭐️⭐️   |   0.5hour    |✅   |
|day27|  [6-4,使用多GPU训练模型](./6-4,使用多GPU训练模型.md)    | ⭐️⭐️   |   0.5hour    |✅  |
|day28|  [6-5,使用TPU训练模型](./6-5,使用TPU训练模型.md)   | ⭐️⭐️   |   0.5hour    |✅  |
|day29| [6-6,使用tensorflow-serving部署模型](./6-6,使用tensorflow-serving部署模型.md) | ⭐️⭐️⭐️⭐️| 1hour |✅   |
|day30| [6-7,使用spark-scala调用tensorflow模型](./6-7,使用spark-scala调用tensorflow模型.md) | ⭐️⭐️⭐️⭐️⭐️|2hour|✅  |
|&nbsp;| [后记：一个吃货和一道菜的故事](./后记：一个吃货和一道菜的故事.md) | ⭐️|0hour|✅  |


```python

```

**2，学习环境**


本书全部源码在jupyter中编写测试通过，建议通过git克隆到本地，并在jupyter中交互式运行学习。

为了直接能够在jupyter中打开markdown文件，建议安装jupytext，将markdown转换成ipynb文件。

**此外，本项目也与和鲸社区达成了合作，可以在和鲸专栏fork本项目，并直接在云笔记本上运行代码，避免环境配置痛苦。** 

🐳和鲸专栏地址：https://www.kesci.com/home/column/5d8ef3c3037db3002d3aa3a0

```python
#克隆本书源码到本地,使用码云镜像仓库国内下载速度更快
#!git clone https://gitee.com/Python_Ai_Road/eat_tensorflow2_in_30_days

#建议在jupyter notebook 上安装jupytext，以便能够将本书各章节markdown文件视作ipynb文件运行
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U jupytext
    
#建议在jupyter notebook 上安装最新版本tensorflow 测试本书中的代码
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -U tensorflow
```

```python
import tensorflow as tf

#注：本书全部代码在tensorflow 2.1版本测试通过
tf.print("tensorflow version:",tf.__version__)

a = tf.constant("hello")
b = tf.constant("tensorflow2")
c = tf.strings.join([a,b]," ")
tf.print(c)
```

```
tensorflow version: 2.1.0
hello tensorflow2
```




```python

```

### 六，鼓励和联系作者 🎈🎈


**如果本书对你有所帮助，想鼓励一下作者，记得给本项目加一颗星星star⭐️，并分享给你的朋友们喔😊!** 

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"Python与算法之美"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![image.png](./data/Python与算法之美logo.jpg)

```python

```
