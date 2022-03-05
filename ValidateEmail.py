import re
regex='^(\w|\_|\.|\-)+[@](\w|\_|\.|\-)+[.]\w{2,3}$'

def check(email):
    if(re.search(regex,email)):
        print("Valid email")
    else:
        print("Not a valid email")

if __name__=='__main__':
    check("garg.nishtha@gmail.com")