/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

template<typename T>
Vec3_t<T> cube_index_to_offset(uint index, uint grid_size)
{
    return Vec3_t<T>(
        index % grid_size,
        (index / grid_size) % grid_size,
        index / (grid_size * grid_size)
    );
}

template<typename T>
std::array<Vec3_t<T>, 8UL> get_subcubes_positions(Vec3_t<T> p, uint s)
{
    T h = static_cast<T>(s) / 2;

    std::array<Vec3_t<T>, 8UL> ret = {{
        { p.x, p.y, p.z },
        { p.x + h, p.y, p.z },
        { p.x, p.y + h, p.z },
        { p.x, p.y, p.z + h },
        { p.x + h, p.y + h, p.z },
        { p.x, p.y + h, p.z + h },
        { p.x + h, p.y, p.z + h },
        { p.x + h, p.y + h, p.z + h }
    }};

    return ret;
}

uint TreeMeshBuilder::decomposeOctree(Vec3_t<float> position, uint size, const ParametricScalarField &field)
{
    uint totalCubesCount = 0;
/*
    Vec3_t<float> S = {
        position.x + float(size) / 2,
        position.y + float(size) / 2,
        position.z + float(size) / 2
    };
    const float mul = sqrt(3.0) / 2;

    if ()
*/
    

    if (size > 1) {
        uint new_size = size / 2;
        std::array<Vec3_t<float>, 8> subcubes = get_subcubes_positions(position, size);
        
        for (auto sc_pos : subcubes) {
            decomposeOctree(sc_pos, new_size, field);
        }

    } else
        totalCubesCount = buildCube(position, field);

    return totalCubesCount;

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    return decomposeOctree(Vec3_t<float>(0.f, 0.f, 0.f), mGridSize, field);
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
#   pragma omp simd reduction(min: value) simdlen(64)
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = value > distanceSquared ? distanceSquared : value;
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.

#   pragma omp critical
    {
        mTriangles.push_back(triangle);
    }
}
