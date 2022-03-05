# Python Program for
# Formatting of Strings

# Default order
String1 = "{} {} {}".format('Happiness', 'In', 'Life')
print("Print String in default order: ")
print(String1)

# Positional Formatting
String1 = "{1} {0} {2}".format('Happiness', 'In', 'Life')
print("\nPrint String in Positional order: ")
print(String1)

# Keyword Formatting
String1 = "{l} {f} {g}".format(g='Happiness', f='In', l='Life')
print("\nPrint String in order of Keywords: ")
print(String1)