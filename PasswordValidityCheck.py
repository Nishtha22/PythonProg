import re
password="R@m@Rama_fortune$"
"""special_char="_@$"

sp_char,lower,upper,no=0,0,0,0
for pas in password:
        if pas in special_char:
            sp_char+=1
        elif pas>='a' and pas<='z':
                lower+=1
        elif pas>='A' and pas<='Z':
                upper+=1
        elif int(pas) in range(10):
                no+=1
print(sp_char)
print(upper)
print(no)
if len(password)>=8 and sp_char>=1 and upper>=1 and no>=1:
    print("Valid Password")
else:
    print("invalid password")"""

#Alternative method ---Regular expression
flag=0
while True:
    if len(password)<8:
        flag=-1
        break
    elif not re.search("[A-Z]",password):
        flag=-1
        break
    elif not re.search("[a-z]",password):
        flag=-1
        break
    elif not re.search("[0-9]",password):
        flag=-1
        break
    elif not re.search("[_@$]",password):
        flag=-1
        break
    else:
        flag=0
        print("Valid password")
        break
if flag==-1:
    print("Not a valid password")


