def ischeck(x):
    print("Check the difference between 'is' and '='")
    print("  {0}   is   [] : ".format(x), x is [])
    print("  {0} is not [] : ".format(x), x is not [])
    print("  {0}   ==   [] : ".format(x), x == [])
    print("  {0}   !=   [] : ".format(x), x != [])

ischeck(3)
ischeck(None)
ischeck([])
"""
Results:
Check the difference between 'is' and '='
  3   is   [] :  False
  3 is not [] :  True
  3   ==   [] :  False
  3   !=   [] :  True
Check the difference between 'is' and '='
  None   is   [] :  False
  None is not [] :  True
  None   ==   [] :  False
  None   !=   [] :  True
Check the difference between 'is' and '='
  []   is   [] :  False
  [] is not [] :  True
  []   ==   [] :  True
  []   !=   [] :  False
"""

a=10**3
b=None
c=[]
d=[]
print("{0:^10}: ".format("id(3)"),id(1000))
print("{0:^10}: ".format("id(a)"),id(a))
print("{0:^10}: ".format("id(None)"),id(None))
print("{0:^10}: ".format("id(b)"),id(b))
print("{0:^10}: ".format("id([])"),id([]))
print("{0:^10}: ".format("id(c)"),id(c))
print("{0:^10}: ".format("id(d)"),id(d))
"""
Results:
id(3): 140462091196928
id(a): 140462091196928
id(None): 140462090894544
id(b): 140462090894544
id([]): 140462058569800
id(c): 140462058462280
id(d): 140462058569864
"""