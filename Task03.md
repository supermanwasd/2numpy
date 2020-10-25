## Task03数组的操作
### 更改形状
1. numpy.ndarray.shape 表示数组的维度,返回一个元组,这个元组的长度就是维度的数目,即 ndim 属性(秩)。
2. numpy.ndarray.flat 将数组转换为一维的迭代器,可以用for访问数组每一个元素。
3. numpy.ndarray.flatten([order='C']) 将数组的副本转换为一维数组,并返回。
a. order:'C' -- 按行,'F' -- 按列,'A' -- 原顺序,'k' -- 元素在内存中的出现顺序。(简记)
b. order:{'C / F,'A,K},可选使用此索引顺序读取a的元素。'C'意味着以行大的C风格顺序对元素进行索引,最后一个轴索引会更改F
表示以列大的Fortran样式顺序索引元素,其中第一个索引变化最快,最后一个索引变化最快。请注意,'C'和'F'选项不考虑基础数组的
内存布局,仅引用轴索引的顺序.A'表示如果a为Fortran,则以类似Fortran的索引顺序读取元素在内存中连续,否则类似C的顺序。“ K”
表示按照步序在内存中的顺序读取元素,但步幅为负时反转数据除外。默认情况下,使用Cindex顺序。
4. numpy.ravel(a, order='C')返回视图。
5. numpy.reshape(a, newshape[, order='C']) 在不更改数据的情况下为数组赋予新的形状。
【例】 reshape() 函数当参数 newshape = -1 时,表示将数组降为一维。
### 数组转置
1. numpy.transpose(a, axes=None).
2. numpy.ndarray.T 和上面一样.
### 更改维度
1. numpy.newaxis = None None 的别名,对索引数组很有用。
很多工具包在进行计算时都会先判断输入数据的维度是否满足要求,如果输入数据达不到指定的维度时,可以使用 newaxis 参数来增加一个维度。
2. numpy.squeeze(a, axis=None) 从数组的形状中删除单维度条目,即把shape中为1的维度去掉。
a. a 表示输入的数组;
b. axis 用于指定需要删除的维度,但是指定的维度必须为单维度,否则将会报错;
在机器学习和深度学习中,通常算法的结果是可以表示向量的数组(即包含两对或以上的方括号形式[[]]),如果直接利用这个数组进行画图可能显示界面为空(见后面的示例)。我们可以利用 squeeze() 函数将表示向量的数组转换为秩为1的数组,这样利用 matplotlib 库函数画图时,就可以正常的显示结果了。
### 数组组合
1. numpy.concatenate((a1, a2, ...), axis=0, out=None)
2. numpy.stack(arrays, axis=0, out=None)
沿着新的轴加入一系列数组(stack为增加维度的拼接)。
3. numpy.vstack(tup)
4. numpy.hstack(tup)
hstack(),vstack() 分别表示水平和竖直的拼接方式。在数据维度等于1时,比较特殊。而当维度大于或等于2时,它们的作用相当于 concatenate ,用于在已有轴上进行操作。
### 数组拆分
1. numpy.split(ary, indices_or_sections, axis=0)
2. numpy.vsplit(ary, indices_or_sections)
垂直切分是把数组按照高度切分
3. numpy.hsplit(ary, indices_or_sections)
水平切分是把数组按照宽度切分。
### 数组平铺
1. numpy.tile(A, reps)
将原矩阵横向、纵向地复制
2. numpy.repeat(a, repeats, axis=None)
a. axis=0 ,沿着y轴复制,实际上增加了行数。
b. axis=1 ,沿着x轴复制,实际上增加了列数。
c. repeats ,可以为一个数,也可以为一个矩阵。d. axis=None 时就会flatten当前矩阵,实际上就是变成了一个行向量。
### 添加和删除元素
1. numpy.unique(ar, return_index=False, return_inverse=False,return_counts=False, axis=None)
