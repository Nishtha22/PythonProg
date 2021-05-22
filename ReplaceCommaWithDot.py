my_string="New York, Dallas, Austin, San.Jose"
str1=my_string.replace(',',"comma")
str2=str1.replace('.',',')
str3=str2.replace("comma",'.')
print(str3)
