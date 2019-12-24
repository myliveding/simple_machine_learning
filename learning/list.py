# python3 test list use

list1 = ['Google', 'Runoob', 1997, 2000];
list2 = [1, 2, 3, 4, 5, 6, 7];

print("list1[0]: ", list1[0])
print("list2[1:5]: ", list2[1:5])

print(4 == len(list1))
print(len(list2))
print(1997 in list1)

list1.append("5555")
for x in list1: print(x, end=" ")

L = ['Google', 'Runoob', 'Taobao']
print(L[1:])
print(L[2])
print(L[-2])

...

...
letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
print(len(letter))
print(letter[1:5:3])

n = len(letter)
if (n > 10):
    print(f"List is too long ({n} elements, expected <= 10)")

if (n := len(letter)) > 10:
    print(f"List is too long ({n} elements, expected <= 10)")


# if (mo := re.search(r'(\d+)% discount', advertisement)):
#     print(float(mo.group(1)) / 100.0)