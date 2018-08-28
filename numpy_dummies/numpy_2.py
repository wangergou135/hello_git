import numpy as np

def maximum_usage():
    a=np.array([1,3,4,5])
    b=np.array([2,3,3,6])
    print(np.maximum(a, b))#[2 3 4 6]
    print(np.minimum(a, b))#[1 3 3 5]

def rand_choice_usage():
    print(np.random.choice(
        np.arange(100), 10, replace=False))

def slice_test():
    a=np.arange(0,16).reshape([4,4])
    print(a[np.array([0,1,3]), np.array([0,2,3])])
if __name__ == '__main__':
    #maximum_usage()
    rand_choice_usage()
    slice_test()