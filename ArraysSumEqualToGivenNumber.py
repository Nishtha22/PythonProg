#Two unsorted array are given. We need to find the sum of all pairs which sum up to x
import array as arr


def sum_pairs(arr1,arr2,x):
    print([(k,x-k) for k in arr1 if (x-k) in arr2])

if __name__=="__main__":
    arr1=arr.array('i',[2,1,4,5])
    arr2=arr.array('i',[6,7,8,5,5])
    x=9
    sum_pairs(arr1, arr2, x)
