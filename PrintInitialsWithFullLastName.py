string="Nishtha Garg"
a=string.split(" ")
for st in range(len(a)-1):
    initial=a[st][0]
    print(initial.upper()+".",end="")
print(a[-1][0].upper()+a[len(a)-1][1:])
