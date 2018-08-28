import numpy as np

a = np.array([[0,1,3],[2,6,8]])

print(a[np.logical_and(a>1, a<7)])

def process_pointcloud(point_cloud, cls='Car'):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_point_number = 35
    else:
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.int64)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_point_number = 45

    np.random.shuffle(point_cloud)

    shifted_coord = point_cloud[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int)
    print('Shape:',np.unique(voxel_index, axis=0).shape)
    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    print('P:', point_cloud[:30, 3])
    print('B:', point_cloud.shape, voxel_index.shape)
    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]
    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)
    print('A:', point_cloud.shape, voxel_index.shape, coordinate_buffer.shape)


    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        #print('V:', voxel, tuple(voxel))
        #exit()
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            #print('point:', point)
            #exit()
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)

    #print('NL:',feature_buffer[:, :, :3].shape, feature_buffer[:, :, :3].sum(axis=1, keepdims=True).shape)
    #print(feature_buffer[0,:,:])
    #print(feature_buffer[0,0,:], number_buffer[0], feature_buffer[0:2, 0:2, :3].sum(axis=1, keepdims=True))
    #exit()
    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict

def test_sum_axis():
    a = np.array([[1,2], [3,4]])
    print('SUM:',a, a.sum(axis=1, keepdims=True).shape, a.sum(axis=1, keepdims=False))

def test_sum_axis1():
    a = np.array([[1,2], [3,4]])
    print('SUM:',a, a.sum(axis=1, keepdims=True), a.sum(axis=1, keepdims=False))
if __name__ == '__main__':
    test_sum_axis()
    print('N:',np.unique(np.array([[1,2],[1,2]]), axis=0))
    raw_lidar=np.fromfile(
        '000000.bin', dtype=np.float32).reshape((-1, 4))

    print(raw_lidar.shape)

    voxel = process_pointcloud(raw_lidar)
    for k,v in voxel.items():

        print(k,v.shape)
        #print(v[:30])