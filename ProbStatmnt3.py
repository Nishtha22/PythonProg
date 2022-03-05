#Longest substring without repeating characters
a='abcbbbbbc'

li=list(a)
for i in li:
    if i!=li[li.index(i)+1]:
        continue
    else:


