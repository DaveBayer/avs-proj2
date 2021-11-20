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
/*
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
*/

template<typename T>
Vec3_t<T> cube_center(Vec3_t<T> p, uint s)
{
    T h = static_cast<T>(s) / 2;

    return { p.x + h, p.y + h, p.z + h };
}
/*
uint TreeMeshBuilder::decomposeOctree(Vec3_t<float> pos, uint size, const ParametricScalarField &field)
{
    uint totalTriangles = 0;
    constexpr float half_sqrt_3 = static_cast<float>(sqrt(3.0) / 2.0);

    Vec3_t<float> S = cube_center(pos, size);
    float r = mIsoLevel + half_sqrt_3 * static_cast<float>(size);

    if (evaluateFieldAt(S, field) > r) {
        if (size > 1) {

            for (auto sc : get_subcubes(pos, size)) {
#               pragma omp task shared(totalTriangles) firstprivate(sc, size, field)
                {
                    totalTriangles += decomposeOctree(sc, size / 2, field);
                }
            }

#           pragma omp taskwait

        } else
            totalTriangles = buildCube(pos, field);
    }

    return totalTriangles;
}
*/
/*
uint TreeMeshBuilder::decomposeOctree(Vec3_t<float> pos, uint size, const ParametricScalarField &field)
{
    uint totalTriangles = 0;
    constexpr float half_sqrt_3 = static_cast<float>(sqrt(3.0) / 2.0);
    
    if (size > 1) {
        uint subcube_size = size / 2;
        float r = mIsoLevel + half_sqrt_3 * static_cast<float>(subcube_size);

        for (auto sc : get_subcubes(pos, size)) {
            Vec3_t<float> S = cube_center(sc, subcube_size);
            
            if (!(evaluateFieldAt(S, field) > r)) {
#               pragma omp task shared(totalTriangles) firstprivate(sc, subcube_size, field)
                {
                    totalTriangles += decomposeOctree(sc, subcube_size, field);
                }
            }
            
        }

#       pragma omp taskwait

    } else
        totalTriangles = buildCube(pos, field);

    return totalTriangles;
}

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
*/
template<typename T>
Vec3_t<T> cube_index_to_offset(uint i, uint grid_size)
{
    return Vec3_t<T>(i % grid_size, (i / grid_size) % grid_size, i / (grid_size * grid_size));
}

std::array<uint, 8UL> get_subcubes(uint index, uint size, uint gs)
{
    uint x_shift = size / 2;
    uint y_shift = x_shift * gs;
    uint z_shift = y_shift * gs;

    return std::array<uint, 8UL> {{
        index,
        index + x_shift,
        index + y_shift,
        index + y_shift + x_shift,
        index + z_shift,
        index + z_shift + x_shift,
        index + z_shift + y_shift,
        index + z_shift + y_shift + x_shift
    }};
}

uint TreeMeshBuilder::decomposeOctree(uint index, uint size, const ParametricScalarField &field)
{
    uint totalTriangles = 0;
    constexpr float half_sqrt_3 = static_cast<float>(sqrt(3.0) / 2.0);
    
    if (size > 1) {
        uint subcube_size = size / 2;
        float r = mIsoLevel + half_sqrt_3 * static_cast<float>(subcube_size);

        for (auto sc : get_subcubes(index, size, mGridSize)) {
            Vec3_t<float> S = cube_center(cube_index_to_offset<float>(sc, mGridSize), subcube_size);
            
            if (!(evaluateFieldAt(S, field) > r)) {
#               pragma omp task shared(totalTriangles) firstprivate(sc, subcube_size, field)
                {
                    totalTriangles += decomposeOctree(sc, subcube_size, field);
                }
            }
            
        }

#       pragma omp taskwait

    } else
        totalTriangles = buildCube(cube_index_to_offset<float>(index), field);

    return totalTriangles;
}

uint TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    uint totalTriangles;

#   pragma omp parallel
#   pragma omp master
    {
        totalTriangles = decomposeOctree(0U, mGridSize, field);
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
