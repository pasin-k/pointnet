import numpy as np
from stl import mesh


def stl_to_point(v1, v2, v3, num_points, sampling_mode="weight"):
    """
    Function to convert stl file into point cloud
    https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
    :param v1, v2, v3 : (N,3) ndarrays, vi represent x,y,z coordinates of one vertex
    :param num_points: Number of points we want to sample
    :param sampling_mode: String, type of sampling from triangle, recommend "weight"
    :return: points: numpy array of point cloud
    """
    if not (np.shape(v1)[0] == np.shape(v2)[0] == np.shape(v3)[0]):
        raise ValueError("Size of all three vertex is not the same")
    else:
        print("Number of mesh: %s" % np.shape(v1)[0])
    areas = triangle_area_multi(v1, v2, v3)
    prob = areas / areas.sum()
    print("Prob")
    print(prob)
    if sampling_mode == "weight":
        indices = np.random.choice(range(len(areas)), size=num_points, p=prob)
    else:
        indices = np.random.choice(range(len(areas)), size=num_points)
    print(indices)
    points = select_point_from_triangle(v1[indices, :], v2[indices, :], v3[indices, :])
    return points


def triangle_area_multi(v1, v2, v3):
    """
    Find area of triangle, used for finding weights
    :param v1, v2, v3 : (N,3) ndarrays, vi represent x,y,z coordinates of one vertex
    :return: size of triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def select_point_from_triangle(v1, v2, v3):
    """
    Select one point from each three vertex
    :param v1, v2, v3 : (N,3) ndarrays, vi represent x,y,z coordinates of one vertex
    :return: ndarrays
    """
    n = np.shape(v1)[0]
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]

    w = 1 - (u + v)

    points = (v1 * u) + (v2 * v) + (v3 * w)

    return points


def view_3d_point_cloud(n):
    """
    Visualize 3d point cloud
    :param n: (N,3) ndarray
    :return: None
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    ax.scatter3D(point[:, 0], point[:, 2], point[:, 1], cmap='Greens')


if __name__ == "__main__":
    a = mesh.Mesh.from_file(
        "/home/pasin/Documents/Link to Tooth/Tooth/Model/global_data/stl_data/84101-2/PreparationScan.stl")

    # v1 = np.array([[1, 0, 0], [2, 3, 1], [-2, 5, 0]])
    # v2 = np.array([[0, 0, 0], [2, 4, 2], [0, 5, 4]])
    # v3 = np.array([[0, 1, 0], [2, 4, 1], [2, 5, 0]])

    point = stl_to_point(a.v0, a.v1, a.v2, 100, sampling_mode="weight")
    print(point)
