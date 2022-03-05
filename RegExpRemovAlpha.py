#Program to remove all the alphabets from the string

import re



patt="[0-9]"

str="arer324353rsdsawr324"
s=re.findall(patt,str)     #--------findall function after importing regular expression re
s1="".join(s)              #--------separator joins with list
print(s1)

