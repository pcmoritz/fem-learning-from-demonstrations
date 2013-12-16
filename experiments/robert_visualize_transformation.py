import numpy
from matplotlib import pyplot

ppu = 200 # points per unit
units = 30

N = ppu * units

def visualize_transformation(ax, f, out_of_bounds):
    def plot_transformed_points(f, points):
        num_points = points.shape[0]
        transformed_points = numpy.zeros((num_points, 2))
        for i in range(num_points):
            transformed_points[i] = f(points[i])
            transformed_points[i] = transformed_points[i] + numpy.array([0, -1]) # so the plots don't collide
        ax.plot(transformed_points[:,0], transformed_points[:,1], color='red')

    # plot rows
    for i in range(units):
        points = numpy.zeros((ppu, 2))
        points[:,0] = numpy.linspace(0, 1, ppu)
        points[:,1] = float(i) / (units - 1) * numpy.ones(ppu)
        j = 0
        while (j < ppu) and (not out_of_bounds(points[j])):
            j += 1
        ax.plot(points[:j,0], points[:j,1], color='blue')
        plot_transformed_points(f, points[:j])
        while j < ppu:
            k = j
            while (k < ppu) and (out_of_bounds(points[k])):
                k += 1
            j = k
            while (j < ppu) and (not out_of_bounds(points[j])):
                j += 1
            ax.plot(points[k:j,0], points[k:j,1], color='blue')
            plot_transformed_points(f, points[k:j])

    # plot columns
    for i in range(units):
        points = numpy.zeros((ppu, 2))
        points[:,0] = float(i) / (units - 1) * numpy.ones(ppu)
        points[:,1] = numpy.linspace(0, 1, ppu)
        j = 0
        while (j < ppu) and (not out_of_bounds(points[j])):
            j += 1
        ax.plot(points[:j,0], points[:j,1], color='blue')
        plot_transformed_points(f, points[:j])
        while j < ppu:
            k = j
            while (k < ppu) and (out_of_bounds(points[k])):
                k += 1
            j = k
            while (j < ppu) and (not out_of_bounds(points[j])):
                j += 1
            ax.plot(points[k:j,0], points[k:j,1], color='blue')
            plot_transformed_points(f, points[k:j])
