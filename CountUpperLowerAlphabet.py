string="geeksForgeeks"
l,u=0,0
#Approach 1
# for chars in string:
#        if chars>='a' and chars<='z':
#            l=l+1
#        if(chars>='A' and chars<='Z'):
#            u=u+1

#Approach 2
for i in range(len(string)):
    if (ord(string[i])>=97) and (ord(string[i])<=122):
        l += 1
    elif (ord(string[i])>=65) and (ord(string[i])<=90):
        u+=1
print('lower',l)
print('upper',u)

