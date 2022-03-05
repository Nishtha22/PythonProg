#Input l is given; l2 is the output given that l is in ascending order
l=[1,2,3,4,7,8,9,13,14,15]
#l2=["1->4","7->9","13->15"]
l2=[]
min=l[0]
for i in l:
    if (l.index(i)+1)!=len(l) and l[l.index(i)+1]==i+1:
      continue
    else:
        max=l[l.index(i)]
        l2.append(str(min) + "->" + str(max))
        if l.index(i)+1!=len(l):
          min=l[l.index(i)+1]
for j in l2:
     print(j)


