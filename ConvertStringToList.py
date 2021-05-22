#Program to convert string to list- split() function gives the result in the form of list
"""Input_string="Geeks for Geeks"
b=Input_string.split(" ")
print(type(b))"""



#Program to convert a single string to list

Input_string="ABCD"
string=[]
for i in range(len(Input_string)):
    string.append(Input_string[i])
print(string)


#ASK-------------------------
def Convert(string):
    list1=[]
    list1[:0]=string
    return list1
# Driver code
str1="ABCD"
print(Convert(str1))
