/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  David Bayer <xbayer09@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    2021/11/16
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <bitset>
#include <omp.h>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

template<typename T>
std::array<Vec3_t<T>, 8UL> get_subcubes(Vec3_t<T> p, uint s)
{
    T h = static_cast<T>(s) / 2;

    return std::array<Vec3_t<T>, 8UL> {{
        { p.x, p.y, p.z },
        { p.x + h, p.y, p.z },
        { p.x, p.y + h, p.z },
        { p.x, p.y, p.z + h },
        { p.x + h, p.y + h, p.z },
        { p.x, p.y + h, p.z + h },
        { p.x + h, p.y, p.z + h },
        { p.x + h, p.y + h, p.z + h }
    }};
}

template<typename T>
Vec3_t<T> cube_center(Vec3_t<T> p, uint s)
{
    T h = static_cast<T>(s) / 2;

    return { p.x + h, p.y + h, p.z + h };
}

uint TreeMeshBuilder::decomposeOctree(Vec3_t<float> pos, uint size, const ParametricScalarField &field)
{
    uint totalTriangles = 0;
    uint half_size = size / 2;

    Vec3_t<float> S = { pos.x + half_size, pos.y + half_size, pos.z + half_size };
    float r = mIsoLevel * static_cast<float>(size) * sqrt(3.0) / 2.0;

    if (evaluateFieldAt(S, field) > r) {
        if (size > 1) {

#           pragma omp task shared(totalTriangles)
            for (auto sc_pos : get_subcubes(pos, size)) {
                totalTriangles += decomposeOctree(sc_pos, half_size, field);
            }

#           pragma omp taskwait

        } else
            totalTriangles = buildCube(pos, field);
    }

    return totalTriangles;
}

/*
uint TreeMeshBuilder::decomposeOctree(Vec3_t<float> pos, uint size, const ParametricScalarField &field)
{
    uint totalTriangles = 0;
    
    if (size > 1) {
        uint half_size = size / 2;
        float r = mIsoLevel * static_cast<float>(half_size) * sqrt(3.0);

        for (auto sc : get_subcubes(pos, size)) {
            Vec3_t<float> S = cube_center(sc, half_size);
            
            if (evaluateFieldAt(S, field) > r) {
#               pragma omp task shared(totalTriangles)
                totalTriangles += decomposeOctree(sc, half_size, field);
            }
            
        }

#       pragma omp taskwait

    } else
        totalTriangles = buildCube(pos, field);

    return totalTriangles;
}
*/
unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    uint totalTriangles;

#   pragma omp parallel
#   pragma omp master
    {
        totalTriangles = decomposeOctree(Vec3_t<float>(0.f, 0.f, 0.f), mGridSize, field);
    }

    return totalTriangles;
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
