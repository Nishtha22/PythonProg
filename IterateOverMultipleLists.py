#Use of zip() function to iterate multiple lists at the same time
import itertools

#zip() returns an iterator
#zip() will stop once the small list among the lists is exhausted.

a=[1,2,3]
b=['a','b','c']

for (x,y) in zip(a,b):
    print(x,y)

#itertools.zip_longest() will run till the longest list and places None for the unknown value. However a default value can be set as well.

l=[4,5]
m=[7,8,9]

for (x,y) in itertools.zip_longest(l,m, fillvalue=-1):
    print(x,y)