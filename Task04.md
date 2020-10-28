### **将数组a中大于30的值替换为30，小于10的值替换为10。**
```python
b = np.clip(a, a_min=10, a_max=30)
print(b)
```

### **找到一个一维数字数组a中的所有峰值。峰顶是两边被较小数值包围的点。**
```python
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
b1 = np.diff(a)
b2 = np.sign(b1)
b3 = np.diff(b2)

print(b1)  # [ 2  4 -6  1  4 -6  1]
print(b2)  # [ 1  1 -1  1  1 -1  1]
print(b3)  # [ 0 -2  2  0 -2  2]
index = np.where(np.equal(b3, -2))[0] + 1
print(index) # [2 5]
```
### **对于给定的一维数组，计算窗口大小为3的移动平均值。**
```python
np.random.seed(100)
z = np.random.randint(10, size=10)
print(z)
# [8 8 3 7 7 0 4 2 5 2]

def MovingAverage(arr, n=3):
    a = np.cumsum(arr)
    a[n:] = a[n:] - a[:-n]
    return a[n - 1:] / n


r = MovingAverage(z, 3)
print(np.around(r, 2))
# [6.33 6.   5.67 4.67 3.67 2.   3.67 3.  ]
```
### **对一个5x5的随机矩阵做归一化**
```python
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)
```
### **用五种不同的方法去提取一个随机数组的整数部分**
```python
Z = np.random.uniform(0,10,10)

print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))
```
### **考虑一维数组Z，构建一个二维数组，其第一行为（Z [0]，Z [1]，Z [2]），随后的每一行都移位1（最后一行应为（Z [ -3]，Z [-2]，Z [-1]）**
```python
from numpy.lib import stride_tricks
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print (Z)
```
### **考虑两组点集P0和P1去描述一组线(二维)和一个点p,如何计算点p到每一条线 i (P0[i],P1[i])的距离？**
```python
def distance(P0,P1,p):
    A=-1/(P1[:,0]-P0[:,0])
    B=1/(P1[:,1]-P0[:,1])
    C=P0[:,0]/(P1[:,0]-P0[:,0])-P0[:,1]/(P1[:,1]-P0[:,1])
    return np.abs(A*p[:,0]+B*p[:,1]+C)/np.sqrt(A**2+B**2)

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))

print (distance(P0, P1, p))
```
### **画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)？**
```python
from matplotlib import pyplot as plt
x = np.arange(0, 3*np.pi, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1)
plt.plot(x, y2)
```

### **减去矩阵每一行的平均值 ？**
```python
X = np.random.rand(5, 10)
# 新
Y = X - X.mean(axis=1, keepdims=True)
# 旧
Y = X - X.mean(axis=1).reshape(-1, 1)
print(Y)
```

### **进行概率统计分析 ？**
```python
arr1 = np.random.randint(1,10,10)
arr2 = np.random.randint(1,10,10)

print("arr1的平均数为:%s" %np.mean(arr1))
print("arr1的中位数为:%s" %np.median(arr1))
print("arr1的方差为:%s" %np.var(arr1))
print("arr1的标准差为:%s" %np.std(arr1))
print("arr1,arr的相关性矩阵为:%s" %np.cov(arr1,arr2))
print("arr1,arr的协方差矩阵为:%s" %np.corrcoef(arr1,arr2))
```

### **获取a和b元素匹配的位置。**
```python
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
mask = np.equal(a, b)

# 方法1
x = np.where(mask)
print(x)  # (array([1, 3, 5, 7], dtype=int64),)

# 方法2
x = np.nonzero(mask)
print(x)  # (array([1, 3, 5, 7], dtype=int64),)
```

### **获取5到10 之间的所有元素。**
```python
a = np.array([2, 6, 1, 9, 10, 3, 27])
mask = np.logical_and(np.greater_equal(a, 5), np.less_equal(a, 10))

# 方法1
x = np.where(mask)
print(a[x])  # [ 6  9 10]

# 方法2
x = np.nonzero(mask)
print(a[x])  # [ 6  9 10]

# 方法3
x = a[np.logical_and(a >= 5, a <= 10)]
print(x)  # [ 6  9 10]
```

### **何对布尔值取反，或者原位(in-place)改变浮点数的符号(sign)？**
```python
Z = np.array([0,1])
print(Z)
np.logical_not(Z, out=Z)
```

### **找出数组中与给定值最接近的数**
```python
Z=np.array([[0,1,2,3],[4,5,6,7]])
print(Z)
z=5.1
np.abs(Z - z).argmin()
print(Z.flat[np.abs(Z - z).argmin()])
```