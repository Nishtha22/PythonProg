#Count and display vowels in a string

my_str="I am a girl"
#output 1--- iaai

vowel="AaEeIiOoUu"

for char in my_str:
    if char in vowel:
        print(char)

#other way to write this is ----- result=[char for char in my_str if char in vowel]

#------------------------------Another output-------------------------------------------------------------
#----------{'a': 2, 'e': 0, 'i': 2, 'o': 0, 'u': 0}------------------------------------------------------

vowels="aeiou"
str=my_str.casefold()
dict={}.fromkeys(vowels,0)

for i in str:
    if i in dict:
        dict[i]+=1
print(dict)