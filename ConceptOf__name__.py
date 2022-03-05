#__name__ is a special variable which evaluates to the name of the current module

# File1.py

print("File1 __name__ = %s" % __name__)

if __name__ == "__main__":
    print("File1 is being run directly")
else:
    print("File1 is being imported")

