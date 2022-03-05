import re
string1="geeksforgeeks"
string2="for"
#solution1
"""if string2 in string1:
    print("A substring")
else:
    print("Not a substring")
"""

#use of find method
"""
if string1.find(string2)!=-1:
    print("A substring")
else:
    print("Not a substring")
"""

#use of regular expression

if re.search(string2,string1):
    print("A substring")
else:
    print("Not a substring")

