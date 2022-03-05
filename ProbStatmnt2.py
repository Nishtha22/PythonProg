# 10+5-30*40/10------input expression
# 10/40*30-5+10------output expression
import re
import array


s="10+5-30*40/10"
x=re.findall('[0-9]+', s)  #creating list1
x1=re.findall('[+*/-]',s)  #creating list2
result=[None]*(len(x)+len(x1)) #creating the empty list of strings having length of list1+list2
result[::-2]=x   #reading the list from back and placing the list1 characters
result[-2::-2]=x1
print(''.join(result))  #converting list back to string





