# python 笔记

## 第二章

- python解析器 - 运行python 程序的程序，代码与硬件间的软件逻辑层
- 源码->字节码->PVM 字节码不是2进制机器指令 pvm是python 虚拟机
- pypy  Rpython 实现的解析器 ，shedskin c++ 转化器 把python 转换成C++ ，冻结二进制 打包分发

## 第三章

* ```#!/user/local/bin/python``` UNIX可执行脚本 `#!/usr/bin/env python` 避免绝对路径 使用 env查找python
* 模块导入的模块只在每次运行时第一次运行，因为导入是个巨大的开销，除非使用reload函数，载入文件最新的版本
* 无论使用import 或者from 导入模块 模块文件中的代码都会执行
* exec可以直接解析源码运行，每次调用都会重新运行

## 第四章

* 列表解析`[row[1] +1 for in M if row[1]%2 ==0]`
* 删除字典 ` for i in list(a.keys()): del a[i]`
* `{1}`  类型是set 集合
* round(2.000,2)以小数点数后多少位输出浮点数,四舍五入

## 第五章

* ^  异或
* ~ 按位取反
* 计算机硬件限制无法精确表现一些值
* 1==2<3 False
* 4/2=2.0 5//2=2
* 复数 2+3j
* int(,16) 16转换成10进制
* 16进制 `hex(), 0xFF eval('0xff') {0:x}.format()`
* 8进制  oct() 0o1 {0:o}.format()
* x.bit_length() 获取二进制长度
* round() 方法返回浮点数x的四舍五入值。
* pyhon 小数对象高精度 decimal  `decimal.getcontext().prec=4`全局精度
* 分数` form fractions imort Fraction x=Fraction(4,6)`
* 集合无序唯一不可变

## 第六章

* 类属于对象而不是变量
*  [垃圾回收](https://juejin.im/post/5b34b117f265da59a50b2fbe)
* is 比较地址值 ==对比数值 数字能使用两种方式比较 因为python机制 内存优化的结果 短字符串也是但是不推荐

## 第七章

* raw字符串抑制转义  但是也不能以单反斜杠结尾
* `print(,end=' ')`
* `'{0:>10} {0:<10} {0:^10} {0:=10}'` 左对齐 右对齐 居中对齐 标记字符后的补充

## 第九章

- 数字通过相对大小进行比较
- 字符串按照字典序 从左到右 一个一个比较 
- 列表和元组从左到右对每部分的内容进行比较
- 数字混合型不能比较
- 字典 比较`sourted(d1.items())<sourted(d2.items())`
- 数字非0为真，其他对象非空为真
- 赋值生成引用而不是拷贝
- python 在对象中检测到循环会打印成[...]而不会陷入无限循环

## 第十一章

* 增强赋值语句 `x+=1` 比通常语句执行快，优化技术会自动选择支持，能在原处修改的就在原处修改，注意可变对象。

## 第十七章

* LEGB `locals -> enclosing function -> globals -> __builtins__` 

  locals 是函数内的名字空间，包括局部变量和形参
  enclosing 外部嵌套函数的名字空间（闭包中常见）
  globals 全局变量，函数定义所在模块的名字空间
  builtins 内置模块的名字空间

## 第二十一章

* python 模块搜索路径 
  * 程序主目录
  * pythonpath的目录
  * 标准链接库的目录
  * 任何.pth文件内容

## 第二十二章

* 模块语句会在首次导入时执行
* 顶层的赋值语句会创建模块属性
* 模块的命名空间能通过`__dict__`或dir()获取
* 模块是个独立的作用域(本地变量就是全局变量)

从此啛啛喳喳啛啛喳喳啛啛喳喳啛啛喳喳表白表白表白表白从此出版保持

* 相对导入适用于只在包内的导入
* 相对导入只是用于from语句
* 相对导入术语含糊不清从此啛啛喳喳啛啛喳喳啛啛喳喳啛啛喳喳表白表白表白表白从此啛啛喳喳啛啛喳喳啛啛喳喳啛啛喳喳表白表白表白表白从此出版b从此啛啛喳喳啛啛喳喳啛啛喳喳啛啛喳喳表白表白表白表白次出场

