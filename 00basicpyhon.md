## Python基础
## 01 Python中的可变对象和不可变对象
#### 01.01 字符串和编码
1. 字符转整数
```
num = ord('a')
```
2. 整数转字符
```
ch = chr(65)
```

##### 什么是可变/不可变对象
不可变对象，该对象所指向的内存中的值不能被改变。当改变某个变量时候，由于其所指的值不能被改变，相当于把原来的值复制一份后再改变，这会开辟一个新的地址，变量再**指向这个新的地址**。

可变对象，该对象所指向的内存中的值可以被改变。变量（准确的说是引用）改变后，实际上是其所指的值直接发生改变，并没有发生复制行为，也没有开辟新的出地址，通俗点说就是**原地改变**。

Python中，数值类型（`int`和`float`）、字符串`str`、元组`tuple`都是不可变类型。而列表`list`、字典`dict`、集合`set`是可变类型。

#### dict
dict key是不可变对象

#### list dict
1. 和list比较，dict有以下几个特点：

查找和插入的速度极快，不会随着key的增加而变慢；
需要占用大量的内存，内存浪费多；

2. 而list相反：
查找和插入的时间随着元素的增加而增加；
占用空间小，浪费内存很少；

## 02 函数
#### 02.2 定义函数
- 定义函数时，需要确定函数名和参数个数；
- 如果有必要，可以先对参数的数据类型做检查；
- 函数体内部可以用return随时返回函数结果；
- 函数执行完毕也没有return语句时，自动return None。
- 函数可以同时返回多个值，但其实就是一个tuple。

#### 02.4 递归优化
- 使用递归函数的优点是逻辑简单清晰，缺点是过深的调用会导致栈溢出。
- 针对尾递归优化的语言可以通过尾递归防止栈溢出。尾递归事实上和循环是等价的，没有循环语句的编程语言只能通过尾递归实现循环。
- Python标准的解释器没有针对尾递归做优化，任何递归函数都存在栈溢出的问题。

`len`只接受字符串输入

## 03  切片
#### 03.01 切片
```python
list tuple string
[from:to:interval]
```
#### 03.02
``` python
def findMinAndMax(L):
    max1 = min1 = 0
    if isinstance(L, Iterable) is False: # whether or not iterable
        print("error: do not iterable")
    for value in L:
        if value > max1:
            max1 = value
            continue
        elif value < min1:
            min1 = value
            continue
        else:
            continue
    return min1, max1
```

#### 03.03 列表表达式
```python
L7 = ['Hello', 'World', 18, 'Apple', None]
L8 = [ key for key in  L7 if isinstance(key, str)]
print(L8)
```

#### 03.04 列表生成式 生成器
```
# 列表生成式
L9 = [x * x for x in range(1,8)]
print(L9)
# 生成器
L10 = (x * x for x in range(1,8))
# print(L10)
for g in L10:
    print(g)
```

包含yield的函数

##### 小结
最难理解的就是generator和函数的执行流程不一样。函数是顺序执行，遇到return语句或者最后一行函数语句就返回。而变成generator的函数，在每次调用next()的时候执行，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行。

#### 03.05 迭代器
我们已经知道，可以直接作用于for循环的数据类型有以下几种：

- 一类是集合数据类型，如list、tuple、dict、set、str等；
- 一类是generator，包括生成器和带yield的generator function。

- Iterable
这些可以**直接作用于for循环**的对象统称为可迭代对象：**Iterable**。
可以使用isinstance()判断一个对象是否是Iterable对象。
Iterable对象存放在内存中，数量是有限的
- Iterator
可以被**next**()函数调用并**不断返回下一个值**的对象称为迭代器：Iterator。

#### 小结
- 凡是可作用于for循环的对象都是Iterable类型；
- 凡是可作用于next()函数的对象都是Iterator类型，它们表示一个惰性计算的序列；
- 集合数据类型如list、dict、str等是Iterable但不是Iterator，不过可以通过iter()函数获得一个Iterator对象。

#### 集合
集合也有一个与列表一样的 pop 方法。 从集合 pop 一个元素时，一个**随机元素**被删除（记住，集合不同于列表，是无序的，所以没有 "最后一个元素"。


Python的for循环本质上就是通过不断调用next()函数实现的。
```python
it = iter([1, 2, 3, 4, 5])
while True:
    try:
        x = next(it)
        print(x)
    except StopIteration as e:
        if e.value is None:
            continue
        print(e.value)
```

## 04 函数式编程
#### 04.01 高阶函数
- 函数本身也可以赋值给变量，即：变量可以指向函数。
- 既然变量可以指向函数，函数的参数能接收变量，那么一个函数就可以接收另一个函数作为参数，这种函数就称之为**高阶函数**

##### 04.01.01 字符串转整数
```python
# char to int
def char2num(s):
     digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
     return digits[s]


#print(char2num('0'))

# num1num2 = num1*10 + num2
def str2int(num1, num2):
    return num1 * 10 + num2


def str2int2(string):
    return reduce(str2int, map(char2num, string))


print(str2int2('123456'))
```


##### 04.01.02 字符串转浮点数
```python
# string to float
def str2float(ff):
    return reduce(str2int, map(char2num, ff.split('.')[0])) + reduce(str2int, map(char2num, ff.split('.')[1])) / 10 ** len(ff.split('.')[1])

print('str2float(\'123.456\') =', str2float('123.456'))
if abs(str2float('123.456') - 123.456) < 0.00001:
    print('测试成功!')
else:
    print('测试失败!')
```

##### 04.01.03 filter
```python
def not_empty(s):
    return s and s.strip()


L13 = list(filter(not_empty, ['A', '', 'B', None, 'C', '  ']))
print(L13)
```

#####  04.01.04 sorted
```python

L14 = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]


def by_name(t):
    return str.lower(t[0])


def by_score(t):
    return t[-1]


L15 = sorted(L14, key=by_name)
L16 = sorted(L14, key=by_score, reverse=True)
```

#### 04.02 返回函数
```python
def lazy_sum(*args):
    def sum():
        all = 0
        for value in args:
            all = all + value
        return all
    return sum


L17 = (1, 2, 3, 3, 4, 5)

f4 = lazy_sum(1, 2, 3, 3, 4, 5)
print(f4())
```
###### 小结
- 我们在函数lazy_sum中又定义了函数sum，并且，**内部函数sum可以引用外部函数lazy_sum的参数和局部变量**，当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中，这种称为“**闭包**（Closure）”的程序结构拥有极大的威力。
- 返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。
- 如果一定要引用循环变量怎么办？方法是再创建一个函数，**用该函数的参数绑定循环变量当前的值**，无论该循环变量后续如何更改，已绑定到函数参数的值不变。本质是内部函数引用外部变量的值。

```python
def count2():
    fs = []

    def f(j):  # bind current loop variable
        def g():
            return j*j
        return g
    for i in range(1, 4):
        fs.append(f(i))
    return fs


f1, f2, f3 = count2()

print(f1())
print(f2())
print(f3())
```


###### 课后习题
```python
def createCounter():
    num = 0

    def counter():
        nonlocal num  # reference outer variable
        num = num + 1
        return num
    return counter


# 测试:
counterA = createCounter()
print(counterA(), counterA(), counterA(), counterA(), counterA()) # 1 2 3 4 5
counterB = createCounter()
if [counterB(), counterB(), counterB(), counterB()] == [1, 2, 3, 4]:
    print('测试通过!')
else:
    print('测试失败!')
```


#### 04.03 匿名函数
匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果。
```python
def is_odd(n):
    return n % 2 == 1


L19 = list(filter(is_odd, range(1, 20)))


is_odd2 = lambda x : x%2 == 1


L20 = list(filter(lambda x : x%2 == 1, range(1,20)))
L21 = list(filter(is_odd2, range(1,20))) # 匿名函数
print(L20)
print(L21)


def build1(x, y):
    return lambda: x * x + y * y # 返回匿名函数


L22 = build1(1,2)
print(L22)
```

#### 04.03 装饰器

现在，假设我们要**增强now**()函数的功能，比如，在函数调用前后自动打印日志，但又不希望修改now()函数的定义，这种在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）。

```python
print("---------------------%s------------------------\n" %("decorator"))


# print(build1.__name__)

# two layers of nested
def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper


@log
def now():
     print('2015-3-25')


print(now())
print("func name: %s" % now.__name__)


print("---------------------%s------------------------\n" %("decorator"))

# three layers of nested
def log2(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print('call: %s %s():' %(text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator


@log2('execute')
def now2():
     print('2015-3-25')


print(now2())
print(now2.__name__)
```

###### console
```python
---------------------decorator------------------------

call now():
2015-3-25
None
func name: wrapper
---------------------decorator------------------------

call: execute now2():
2015-3-25
None
wrapper
```

##### 课后练习
```python
def metric(fn):
    def wrapper(*args, **kw):
        start = time.time()
        print('call start: %s %s():' % (fn.__name__, start))
        num = fn(*args, **kw)
        end = time.time()
        print('call end: %s %s():' % (fn.__name__, end))
        print('%s executed in %s ms' % (fn.__name__, end - start))
        return num
    return wrapper


# 测试
@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y;

@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;

f = fast(11, 22)
s = slow(11, 22, 33)

print(f)
print(s)

if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')
```

#### 04.04 偏函数
简单总结`functools.partial`的作用就是，把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。

```python
int2 = functools.partial(int, base=2)
print(int2('1000000'))

int10 = functools.partial(int, base=8) # 默认输入八进制
print(int10('12'))
```

## 06 面向对象编程
特殊方法“__init__”前后分别有两个下划线！！！

#### 06.01 类和实例
#### 06.02 访问限制
```python
class Student(object):
    def __init__(self, name, score):
        self.__name = name
        self.__score = score

    def get_grade(self):
        if self.__score >= 90:
            return 'A'
        elif self.__score >= 60:
            return 'B'
        else:
            return 'C'

    def print_student(self):
        print("name%s, score=%s" %(self.__name, self.__score))

    def get_score(self):
        return self.__score

    def get_name(self):
        return self.__name

    def set_score(self, score):
        if 0<= score <= 100 :
            self.__score = score
        else:
            raise  ValueError("score should range 0-100")

    def set_name(self, name):
        self.__name = name

student = Student('Oppo', 23)

student.print_student()
score_level = student.get_grade()
print(score_level)

# debug throw exception
# student.set_score(-1)
```

#### 06.03 继承与多态
```python
class Animal(object):
    def run(self):
        print('Animal is running...')

class Dog(Animal):
    def run(self):
        print('Dog is running...')

class Cat(Animal):
    def run(self):
        print('Cat is running...')

animal = Animal()
animal.run()

animal = Dog()
animal.run()

animal = Cat()
animal.run()

```

#### 06.04 获取对象信息
- isinstance()判断的是一个对象是否是该**类型本身**，或者位于该类型的**父继承链上**.
- 能用type()判断的基本类型也可以用isinstance()判断.

```python
class MyObject(object):
    def __init__(self):
        self.x = 9
    def power(self):
        return self.x * self.x

myobject = MyObject()
print(hasattr(myobject, 'x'))
setattr(myobject, 'x', 10)
setattr(myobject, 'y', 13)
print(getattr(myobject, 'y', -1))
if hasattr(myobject, 'power'):
    power = getattr(myobject, 'power', -1)
    print(power())
```

#### 06.05 实例属性和类属性
由于实例属性优先级比类属性高，因此，它会屏蔽掉类的name属性

在编写程序的时候，千万**不要对实例属性和类属性使用相同的名字**，因为相同名称的实例属性将屏蔽掉类属性，但是当你删除实例属性后，再使用相同的名称，访问到的将是类属性。

```python
class Student(object):
    count = 0 # static varible

    def __init__(self, name):
        self.name = name # instance varible
        Student.count += 1
```
## 07 面向对象高级编程
#### 07.01 使用__slots__
- 使用__slots__限制实例的属性
- 用__slots__要注意，__slots__定义的属性仅对当前类实例起作用，对继承的子类是不起作用的


```python
class Teacher(object):
    __slots__ = ('name', 'age', 'score')

t = Teacher()
t.name = "xiaomi"
print(t.name)


# dynamic bind function for instance

def set_age(self, age):
    self.age = age

# after init __slots__ , don't allow bind set_age
# t.set_age = MethodType(set_age, t)
#
# t.set_age(23)
# print(t.age)
#

t2 = Teacher()  # unbind on other instance


# t2.set_age(12)
# print(t2.age)


def set_score(self, score):
    self.score = score


# bind function on object
Teacher.set_score = set_score
t2.set_score(89)
print(t2.score)

class GraduateTeacher(Teacher):
    pass

# __slots__ is noeffective to child class
g = GraduateTeacher()
g.gender = 'M'

print(g.gender)
```

#### 07.02 使用@property
- 既能检查参数，又可以用类似属性这样简单的方式来访问类的变量

```python
class Screen(object):
    def __init__(self):
        pass

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @width.setter
    def height(self, value):
        self._height= value

    @property
    def resolution(self):
        return self._width * self._height

# 测试:
s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')
```

#### 07.03 多重继承
**MixIn**的目的就是给一个类增加多个功能，这样，在设计类的时候，我们优先考虑**通过多重继承来组合多个MixIn的功能**，而不是设计多层次的复杂的继承关系。
```python
class Alpaca(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return "Alpaca name:%s " % self.name
    def __repr__(self):
        return "Alpaca repr name:%s " % self.name

alpaca = Alpaca("mie")
```

#### 07.04 续
#### 07.05 使用枚举类
```python
from enum import Enum, unique


Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

for name, member in Month.__members__.items():
    print(name, member.value)

@unique
class Gender(Enum):
    Male = 0
    Female = 1

class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender


# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')

print(Gender.Male.value)
print(Gender(0).value)
```

#### 07.06 使用元类
要创建一个class对象，type()函数依次传入3个参数：

1. class的名称；
2. 继承的父类集合，注意Python支持多重继承，如果只有一个父类，别忘了tuple的单元素写法；
3. class的方法名称与函数绑定，这里我们把函数fn绑定到方法名hello上。

```python
print("---------------------%s------------------------\n" % ("metaclass"))
from plat import Plat

pl = Plat()
pl.plat()

print(type(Plat)) # class type is Type
print(type(pl)) # instance type is Class Type


print("---------------------%s------------------------\n" % ("type create new class"))

def fn(self, name="pine"):
    print("plat: %s" % name)

Plat2 = type("Plat", (object,), dict(pp=fn))

pl2 = Plat2()
pl2.pp()
```

## 08 错误、调试和测试
#### 08.01 错误处理
```python
try:
    10/0
except ZeroDivisionError as e:
    print(e)
finally:
    print("done")
```
##### 断言


## 20 异步IO
#### 20.01 协程

**协程**看上去也是子程序，但执行过程中，在子程序内部可中断，然后转而执行别的子程序，在适当的时候再返回来接着执行。

看起来A、B的执行有点像多线程，但协程的特点在于是一个线程执行，那和多线程比，协程有何优势？

最大的优势就是协程极高的**执行效率**。因为子程序切换不是线程切换，而是由程序自身控制，因此，**没有线程切换的开销**，和多线程比，线程数量越多，协程的性能优势就越明显。

第二大优势就是**不需要多线程的锁机制**，因为只有一个线程，也不存在同时写变量冲突，在协程中控制共享资源不加锁，只需要判断状态就好了，所以执行效率比多线程高很多。

因为协程是一个线程执行，那怎么利用多核CPU呢？最简单的方法是多进程+协程，既充分利用多核，又充分发挥协程的高效率，可获得极高的性能。

`Python`的`yield`不但可以返回一个值，它还可以接收调用者发出的参数。
###### 例子
```
def consumer():
    num = 0
    r = ''
    while True:
        num = num + 1
        print("----> %2d" % num)
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'


def produce(c):
    r2 = c.send(None)
    if r2 == '':
        print('c send None return null')
    else:
        print('c send None return: %s' % r2)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

c = consumer()
produce(c)
```

###### 日志
```
---->  1
c send None return null
[PRODUCER] Producing 1...
[CONSUMER] Consuming 1...
---->  2
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 2...
[CONSUMER] Consuming 2...
---->  3
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 3...
[CONSUMER] Consuming 3...
---->  4
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 4...
[CONSUMER] Consuming 4...
---->  5
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 5...
[CONSUMER] Consuming 5...
---->  6
[PRODUCER] Consumer return: 200 OK
```
## 30 留白


## 31 数据可视化
#### 31.01 平方图
```python
import matplotlib.pyplot as plt

# x_values = [1, 2, 3, 4, 5]
# y_values = [1, 4, 9, 16, 25]

x_values = list(range(1,1001))
y_values = [x_ * x_ for x_ in x_values]


# 绘制指定的x,y
# plt.scatter(2, 4, s=200)

# debug
# ll1 = [x_ * x_ for x_ in x_values]

# 绘制一系列的x,y

# edgecolors='none' 删除数据的的轮廓
# c=y_values, cmap=plt.cm.Greens 根据y值设置颜色
plt.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Greens, edgecolors='none', s=10)

# 设置图表标题，并给坐标轴加上标签
plt.title("Square Numbers", fontsize=24)
plt.xlabel("value", fontsize=14)
plt.ylabel("square of value", fontsize=14)

plt.tick_params(axis='both', which='major', labelsize=10)

# 设置每个坐标轴的取值范围
plt.axis([0, 1000, 0, 1100000])

# plt.show()
plt.savefig('squares_plot.png', bbox_inches='tight')
```
##### 程序执行结果
![squares_plot](https://i.imgur.com/Cw8qjsg.png)

#### 31.02 概率随机分布图
```python
import matplotlib.pyplot as plt

from random_walk import RandomWalk

while True:
    # 创建一个RandomWalk实例，并将其包含的点都绘制出来
    rw = RandomWalk()

    rw.fill_walk()

    plt.figure(dpi=128, figsize=(10, 6))

    point_numbers = list(range(rw.num_points))
    '''
    # 随机点的轨迹
    plt.plot(rw.x_values, rw.y_values, linewidth=1)
    '''

    # 随机点的分布
    plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Greens, edgecolors='none', s=10)

    # 突出起点和终点
    plt.scatter(rw.x_values[0], rw.y_values[0], c='green', edgecolors='none', s=100)
    plt.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none', s=100)

    # 隐藏坐标轴
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

    # plt.show()
    plt.savefig('RandomWalk.svg', bbox_inches='tight')

    # keep_running =  input('\ny or n:')
    # if keep_running == 'n':
    #     print('good bye')
    #     break
    break
```

###### 程序执行结果
![point_random_distribution](https://i.imgur.com/HTOt3jd.png)

##### 31.03 掷色子
```python
from die import Die
import pygal

die = Die(6)
die2 = Die(6)
die3 = Die(6)

# results = []
# for roll_num in range(50000):
#     result = die.roll() + die2.roll()
#     results.append(result)
# print(results)

results = [die.roll() * die2.roll() * die3.roll() for roll_num in range(10000)]

# frquencies = []
max_num = die.num_sides * die2.num_sides * die3.num_sides
# for value in range(2, max_num + 1):
#     frquency = results.count(value)
#     frquencies.append(frquency)

frquencies = [ results.count(value) for value in range(1, max_num + 1)]

hist = pygal.Bar()

hist.title = "Result of rolling one D6 1000 times"
hist.x_labels = [ str(x) for x in range(1, max_num + 1)]
hist.x_title = 'Result'
hist.y_title = "Frequency of Result"

hist.add('D6 + D6', frquencies)
hist.render_to_file('die_visual2.svg')
# print(frquencies)

```
###### 程序执行结果
![throw_shuazi](https://i.imgur.com/nSP7GM5.png)