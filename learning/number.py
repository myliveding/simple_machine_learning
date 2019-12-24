# 数字类型实践

'''
1、Python可以同时为多个变量赋值，如a, b = 1, 2。
2、一个变量可以通过赋值指向不同类型的对象。
3、数值的除法包含两个运算符：/ 返回一个浮点数，// 向下取接近除数的整数。
4、在混合计算时，Python会把整型转换成为浮点数。
'''
import self as self

print(5 + 4)  # 加法
print(4.3 - 2) # 减法
print(3 * 7)  # 乘法
print(2 / 4)  # 除法，得到一个浮点数
print(2 // 4) # 除法，得到一个整数
print(17 % 3) # 取余
print(2 ** 5) # 乘方

a, b, c, d = 20, 5.5, True, 4+3j
print(type(a), type(b), type(c), type(d))
#<class 'int'> <class 'float'> <class 'bool'> <class 'complex'>
print(isinstance(a, int))
print(isinstance(d, int))


# 开始验证 类的判断
class A:
 pass

class B(A):
 pass

print(isinstance(A(), A))
print(type(A()) == A)
print(isinstance(B(), A))
print(type(B()) == A)


class Solution:
    def NumberOf1(self, n):
        # write code here
        cnt = 0
        if n<0:
            n = n & 0xffffffff
        while n:
            cnt+=1
            n = (n-1) & n
        return cnt

    def NumberOf(self, n):
        # write code here
        print(bin(n))
        return bin(n).count("1")


print(Solution.NumberOf1(self, 100))
print(Solution.NumberOf(self, 100))

