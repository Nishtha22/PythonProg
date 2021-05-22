"""
#Output is ['','','']
input_string="This is my practice"
vowels="AaEeIiOoUu"
li=[]
for vow in input_string:
    if vow in vowels:
        li.append(vow)
print(len(li))
print(li)

#Another way to write the above code is
li=[vow for vow in input_string if vow in vowels]
print(len(li))
print(li)"""

#Suppose the output is in the form of dictionary in which every vowel has to be counted and placed as key-value pair

input_string="This is my practice"
vowels="aeiou"
input_string1=input_string.casefold()
print(input_string)
print(input_string1)

count={}.fromkeys(vowels,0)
for character in input_string1:
    if character in vowels:
        count[character]+=1
print(count)
