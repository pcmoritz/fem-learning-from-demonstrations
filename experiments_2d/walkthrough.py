def main():
    import meshpy.triangle as triangle

    points = [ (1,1),(-1,1),(-1,-1),(1,-1)]

    def round_trip_connect(start, end):
      result = []
      for i in range(start, end):
        result.append((i, i+1))
      result.append((end, start))
      return result

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(points)-1))

    mesh = triangle.build(info, max_volume=1e-3, min_angle=25)

    f = open('output.xml', 'w')

    f.write("""
        <?xml version="1.0" encoding="UTF-8"?>

        <dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">
          <mesh celltype="triangle" dim="2">
            <vertices size="%d">
        """ % len(mesh.points))

    for i, pt in enumerate(mesh.points):
      f.write('<vertex index="%d" x="%g" y="%g"/>' % (
              i, pt[0], pt[1]))

    f.write("""
        </vertices>
        <cells size="%d">
        """ % len(mesh.elements))

    for i, element in enumerate(mesh.elements):
      f.write('<triangle index="%d" v0="%d" v1="%d" v2="%d"/>' % (
              i, element[0], element[1], element[2]))

    f.write("""
            </cells>
          </mesh>
        </dolfin>
        """)

from dolfin import *

main()
