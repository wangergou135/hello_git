import numpy as np


#np.where(condition, x, y)
def where_usage_1():
    aa = np.arange(10)
    print(np.where(aa, 1, -1))
    print(np.where(aa>5,1,-1))

#np.where(condition)
def where_usage_2():
    a = np.array([2, 4, 6, 8, 10])
    print(np.where(a > 5))  # 返回索引
    a = np.arange(27).reshape(3, 3, 3)
    print(np.where(a > 5))#输出每个元素的对应的坐标，因为原数组有三维，所以tuple中有三个数组。

def any_usage():
    a = np.array([1, 3, 5])
    b = a.copy()
    print((a == b).any())
    b[0] = 9
    print((a == b).any())
    b = [2,4,6]
    print((a == b).any())
    c = np.array([[0,2],[0,3]])
    print(c.any(axis=0))

def all_usage():
    a = np.array([1,3,5])
    b = a.copy()
    print( (a==b).all())
    b[0] = 9
    print((a == b).all())

    c = np.array([[0, 2], [0, 3]])
    print(c.all(axis=1))

def stack_usage():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print(np.stack([a, b], axis=0))
    '''
    [[1 2 3]
    [4 5 6]]
    '''
    print(np.stack([a, b], axis=1))
    '''
    [[1 4]
     [2 5]
     [3 6]]
    '''

    print(np.stack([a.reshape([-1, 1]), b.reshape([-1, 1])], axis=0))
    '''
    [[[1]
      [2]
      [3]]
    
     [[4]
      [5]
      [6]]]
    '''
    print(np.stack([a.reshape([-1, 1]), b.reshape([-1, 1])], axis=1).flatten())
    '''
    [[[1]
      [4]]
    
     [[2]
      [5]]
    
     [[3]
      [6]]]
    '''

def vstack_usage():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print(np.vstack([a, b]))
    '''
    [[1 2 3]
    [4 5 6]]
    '''

    print(np.vstack([a.reshape([-1, 1]), b.reshape([-1, 1])]))
    '''
    [[1]
     [2]
     [3]
     [4]
     [5]
     [6]]
    '''

def hstack_usage():
    a = np.array([1,2,3])
    b = np.array([4,5,6])

    print(np.hstack([a,b]))
    #[1 2 3 4 5 6]

    print(np.hstack([a.reshape([-1,1]), b.reshape([-1,1])]))
    '''
    [[1 4]
     [2 5]
     [3 6]]
    '''
if __name__ == '__main__':
    stack_usage()
    a = np.array([1, 2, 3]).reshape([-1,1]).reshape([-1,])#[1,2,3]
    #print(a)
