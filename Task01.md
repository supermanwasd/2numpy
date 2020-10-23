## Task01：数据类型及数组创建
#### 1.常量
##### 1.1 空值
```python
import numpy as np

print(np.nan == np.nan) # False
print(np.nan != np.nan) # True

x = np.array([1, 1, 8, np.nan, 10])# [ 1. 1. 8. nan 10.]

y = np.isnan(x)# [False False False True False]

z = np.count_nonzero(y) # 1
```
##### 1.2 无穷大
```python
np.Inf == np.inf == np.infty == np.Infinity == np.PINF #True
```
##### 1.3 π和自然常数
```python
np.pi #3.141592653589793
np.e #2.718281828459045
```
#### 2.数据类型
##### 2.1 常见数据类型
为了加以区分numpy和python原生的数据类型，在numpy类型名称末尾都加了“_”。

| 类型 | 备注 | 说明 |
| --- | --- | --- |
| bool_ = bool8 | 8位 | 布尔类型 |
| int8 = byte | 8位 | 整型 |
| int16 = short | 16位 | 整型 |
| int32 = intc |32位  | 整型 |
| int_ = int64 = long = int0 = intp | 64位 | 整型 |
##### 2.2 创建数据类型
```python
a = np.dtype("b1")
print(a.type)#<class 'numpy.bool_'>
print(a.itemsize)#1

a = np.dtype("i1")
print(a.type)#<class 'numpy.int8'>
print(a.itemsize)#1
```
##### 2.3 数据类型信息
Python的浮点数通常是64位浮点数，几乎等同于np.float64。
在整数类型的行为在整数溢出方面存在显著差异，python的整数可以扩展以容纳任何整数并且不会溢出。
```python
class iinfo(object):
    def __init__(self, int_type):
        pass
    def min(self):
        pass 
    def max(self):
        pass

ii16 = np.iinfo(np.int16)
print(ii16.min) # -32768 
print(ii16.max) # 32767

ii32 = np.iinfo(np.int32)
print(ii32.min) # -32768 
print(ii32.max) # 32767

ff16 = np.finfo(np.float16)
print(ff16.bits) # 16
print(ff16.min) # -65500.0
print(ff16.max) # 65500.0
print(ff16.eps) # 0.000977
```
#### 3.时间日期和时间增量
##### 3.1 datetime64基础
numpy中是datetime64（datetime已被python中的日期库使用）
| 日期单位 | 代码含义 | 时间单位 | 代码含义 |
| --- | --- | --- | --- |
| Y | 年 | h | 小时 |
| M | 月 | m | 分钟 |
| W | 周 | s | 秒 |
| D | 天 | ms | 毫秒 |
```python
a = np.datetime64("2020-03-01")
print(a,a.dtype)#2020-03-01 datetime64[D]

a = np.datetime64("2020-03")
print(a,a.dtype)#2020-03 datetime64[M]

a = np.datetime64("2020-03-01 20:01:05")
print(a,a.dtype)#2020-03-01T20:01:05 datetime64[s]

a = np.datetime64("2020-03-01 20:01")
print(a,a.dtype)#2020-03-01T20:01 datetime64[m]

a = np.datetime64("2020-03-01 20:01:05","Y")
print(a,a.dtype)#2020 datetime64[Y]

#2019-03和2019-03-01是同一个日期
a = np.array(['2020-03', '2020-03-08', '2020-03-08 20:00'], dtype='datetime64')
print(a, a.dtype)#['2020-03-01T00:00' '2020-03-08T00:00' '2020-03-08T20:00'] datetime64[m]
#创建datetime64数组时，如果单位不同意，则一律转化成其中最小的单位。

a = np.arange('2020-08-01', '2020-08-10', dtype=np.datetime64)
print(a)#['2020-08-01' '2020-08-02' '2020-08-03' '2020-08-04' '2020-08-05' '2020-08-06' '2020-08-07' '2020-08-08' '2020-08-09']
#使用arrange()创建datetime64数组，用于生成日期范围。
```
##### 3.2 datetime64和timedelta64 运算
```python
#timedelta64 表示两个 datetime64 之间的差。timedelta64 也是带单位的，并和相减运算中的两个datetime64中的较小的单位保持一致。

c = np.datetime64('2020-03-08') - np.datetime64('2020-03-03 23:00', 'h')
print(c)#97 hours

a = np.datetime64('2020-03') + np.timedelta64(20, 'D')
print(a)#2020-03-21


a = np.timedelta64(1, 'Y')
print(np.timedelta64(a, 'D'))
#Cannot cast NumPy timedelta64 scalar from metadata [Y] to [D] according to the rule 'same_kind'
#生成timedelta64时，年和月无法和其它单位运算（一年有几天，一月有几天都不确定）

dt = datetime.datetime(year=2020, month=6, day=1, hour=20, minute=5, second=30)
dt64 = np.datetime64(dt, 's')
print(dt64, dt64.dtype)#2020-06-01T20:05:30 datetime64[s]

dt2 = dt64.astype(datetime.datetime)
print(dt2, type(dt2))#2020-06-01 20:05:30 <class 'datetime.datetime'>
#np.datetime64和datetime.datetime可以相互转换

```
##### 3.3 datetime64的应用
numpy包含一组"busday"（工作日）功能。
```python
a = np.busday_offset('2020-07-10', offsets=1)
print(a)#2020-07-13
a = np.busday_offset('2020-07-11', offsets=1)
print(a)#Non-business day date in busday_offset
```
#### 4.数组的创建
##### 4.1 依据现有数据来创建ndarray
###### 4.1.1 通过arrar()函数进行创建
```python
a = np.array([0,1,2,3,4])
b = np.array([1,2,3,4,4])
print(a,type(a))#[0 1 2 3 4] <class 'numpy.ndarray'>
print(b,type(b))#[1 2 3 4 4] <class 'numpy.ndarray'>

c = np.array([[11,12,13,14,15],
            [11,12,13,14,15],
            [11,12,13,14,15],
            [11,12,13,14,15],
            [11,12,13,14,15]])
print(c,type(c))
"""
[[11 12 13 14 15]
 [11 12 13 14 15]
 [11 12 13 14 15]
 [11 12 13 14 15]
 [11 12 13 14 15]] <class 'numpy.ndarray'>
 """
d = np.array([[(1.5,2,3),(4,5,6)],
             [(3,2,1),(4,5,6)]])
print(d,type(d))
 """
 [[[1.5 2.  3. ]
  [4.  5.  6. ]]

 [[3.  2.  1. ]
  [4.  5.  6. ]]] <class 'numpy.ndarray'>
 """
```
###### 4.1.2 通过asarray()函数进行创建
array()和asarray()的区别是当数据源是ndarra时，array()仍然会copy出一个副本，占用新的内存，但不改变dtype时asarray()不会。
```python
x = [[1,1,1],[1,1,1],[1,1,1]]
y = np.array(x)
z = np.asarray(x)
x[1][2] = 2

print(x,type(x))#[[1, 1, 1], [1, 1, 2], [1, 1, 1]] <class 'list'>

print(y,type(y))
"""
[[1 1 1]
 [1 1 1]
 [1 1 1]] <class 'numpy.ndarray'>
"""
print(z,type(z))
"""
[[1 1 1]
 [1 1 1]
 [1 1 1]] <class 'numpy.ndarray'>
"""

x = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
y = np.array(x)
z = np.asarray(x)
w = np.asarray(x,dtype=np.int)
x[1][2] = 2
print(x,type(x),x.dtype)
"""
[[1 1 1]
 [1 1 2]
 [1 1 1]] <class 'numpy.ndarray'> int64
"""

print(y,type(y),y.dtype)
"""
[[1 1 1]
 [1 1 1]
 [1 1 1]] <class 'numpy.ndarray'> int64
"""

print(z,type(z),z.dtype)
"""
[[1 1 1]
 [1 1 2]
 [1 1 1]] <class 'numpy.ndarray'> int64
"""
```
###### 4.1.3 通过fromfunction()函数进行创建
```python
def f(x, y):
    return 10 * x + y

x = np.fromfunction(f, (5, 4), dtype=int)
print(x)
"""
[[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]
"""

x = np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
print(x)
"""
[[ True False False]
 [False  True False]
 [False False  True]]
"""
```
##### 4.2 依据ones和zeros填充方式
机器学习中常做的一件事就是初始化参数，需要用常数值后者随机值来创建一个固定大小的矩阵。
###### 4.2.1 零数组
zeros()函数：返回给定形状和类型的零数组
zeros_like()函数：返回与给定数组形状和类型相同的零数组
###### 4.2.2 1数组
ones()函数：返回给定形状和类型的1数组。
ones_like()函数：返回与给定数组形状和类型相同的1数组。
###### 4.2.3 空数组
empty() 函数：返回一个空数组，数组元素为随机数。
empty_like 函数：返回与给定数组具有相同形状和类型的新数组。
###### 4.2.4 单位数组
eye() 函数：返回一个对角线上为1，其它地方为零的单位数组。
identity() 函数：返回一个方的单位数组。
###### 4.2.5 对角数组
diag() 函数：提取对角线或构造对角数组。
###### 4.2.6 常数数组
full() 函数：返回一个常数数组。
full_like() 函数：返回与给定数组具有相同形状和类型的常数数组。
##### 4.3 利用数值范围来创建ndarray
arange() 函数：返回给定间隔内的均匀间隔的值。
linspace() 函数：返回指定间隔内的等间隔数字。
logspace() 函数：返回数以对数刻度均匀分布。
numpy.random.rand() 返回一个由[0,1)内的随机数组成的数组。
##### 4.4 结构数组的创建
结构数组，首先需要定义结构，然后利用 np.array() 来创建数组，其参数 dtype 为定义的结构。
###### 4.4.1 利用字典来定义结构
```python
personType = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['U30', 'i8', 'f8']})

a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))#[('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)] <class 'numpy.ndarray'>
```
###### 4.4.2 利用多个元组的列表来定义结构
```python
personType = np.dtype([('name', 'U30'), ('age', 'i8'), ('weight', 'f8')])
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>
```
#### 5.数组的属性
numpy.ndarray.ndim用于返回数组的维数（轴的个数）也称为秩，一维数组的秩为 1，二维数组的秩为 2，以此类推。
numpy.ndarray.shape表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 ndim 属性(秩)。
numpy.ndarray.size数组中所有元素的总量，相当于数组的shape中所有元素的乘积，例如矩阵的元素总量为行与列的乘积。
numpy.ndarray.dtype ndarray 对象的元素类型。
numpy.ndarray.itemsize以字节的形式返回数组中每一个元素的大小。

#### 作业
##### 什么是numpy？
一个Python的科学计算包
##### 如何安装numpy？
pip install numpy
##### 什么是n维数组对象？

##### 如何区分一维、二维、多维？
使用numpy.ndarray.ndim判断
##### 以下表达式运行的结果分别是什么?
```python

0 * np.nan #nan
np.nan == np.nan #False
np.inf > np.nan #False
np.nan - np.nan #nan
0.3 == 3 * 0.1 #False
```
##### 将numpy的datetime64对象转换为datetime的datetime对象。
```python
dt2 = dt64.astype(datetime.datetime)
print(dt2, type(dt2))
#2020-02-25 22:10:10 <class 'datetime.datetime'>
```
##### 给定一系列不连续的日期序列。填充缺失的日期，使其成为连续的日期序列。
```python
dates = np.arange('2020-02-01', '2020-02-10', 2, np.datetime64)
print(dates)
# ['2020-02-01' '2020-02-03' '2020-02-05' '2020-02-07' '2020-02-09']

out = []
for date, d in zip(dates, np.diff(dates)):
    out.extend(np.arange(date, date + d))
fillin = np.array(out)
output = np.hstack([fillin, dates[-1]])
print(output)
```
##### 如何得到昨天，今天，明天的的日期
```python
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
```
##### 创建从0到9的一维数字数组。
```python
def f(x,y):
    return y+1
np.fromfunction(f, (1,9), dtype=int)
```
##### 创建一个元素全为True的 3×3 数组。
```python
x = np.fromfunction(lambda i, j: i * 0== 0, (3, 3), dtype=int)
print(x)
```
##### 创建一个长度为10并且除了第五个值为1的空向量
```python
x = np.empty((1,10))
x[0][4] = 10
```
##### 创建一个值域范围从10到49的向量
```python
x = np.arange(10, 50, 1)
```
##### 创建一个 3x3x3的随机数组
```python
np.random.random((3,3))
```
##### 创建一个二维数组，其中边界值为1，其余值为0
```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```
##### 创建长度为10的numpy数组，从5开始，在连续的数字之间的步长为3。
```python
start = 5
step = 3
length = 10
a = np.arange(start, start + step * length, step)
```
##### 将本地图像导入并将其转换为numpy数组。
```python
import numpy as np
from PIL import Image

img1 = Image.open('test.jpg')
a = np.array(img1)

print(a.shape, a.dtype)
```