from collections import Counter
string1="nishtha"
string2="ahthsin"
"""
if sorted(string1)==sorted(string2):
    print("An anagram")
else:
    print("Not an anagram")
"""

#use of Counter---it counts the occurance of the alphabets

if Counter(string2)==Counter(string1):
    print("An anagram")
else:
    print("Not an anagram")
