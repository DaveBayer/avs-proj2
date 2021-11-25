/**
 * @file    loop_mesh_builder.cpp
 *
 * @author  David Bayer <xbayer09@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    2021/11/16
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "loop_mesh_builder.h"

const size_t LoopMeshBuilder::init_malloc_amount = 100000UL;
const size_t LoopMeshBuilder::realloc_amount = 20000UL;

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree"), size(0UL), capacity(init_malloc_amount)
{
    auto is_power_of_2 = [](uint n) -> bool
    {
        return n != 0U && !(n & (n - 1U));
    };

    if (!is_power_of_2(mGridSize))
        throw GridSizeException();
        
    triangles = static_cast<Triangle_t *>(malloc(capacity * sizeof(Triangle_t)));

    if (triangles == nullptr)
        throw std::bad_alloc();
}

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;

#   pragma omp parallel for reduction(+: totalTriangles) schedule(dynamic) collapse(3)
    for(uint i = 0U; i < mGridSize; i++) {
        for (uint j = 0U; j < mGridSize; j++) {
            for (uint k = 0U; k < mGridSize; k++) {
                totalTriangles += buildCube(Vec3_t<float>(k, j, i), field);
            }
        }
    }

    return totalTriangles;
}

float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
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
/*
void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
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
*/

void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
#   pragma omp critical
    {
        if (size + 1 >= capacity) {
            capacity += realloc_amount;
            triangles = static_cast<Triangle_t *>(realloc(triangles, capacity * sizeof(Triangle_t)));

            if (triangles == nullptr)
                throw std::bad_alloc();
        }

        triangles[size++] = triangle;
    }
}

LoopMeshBuilder::~LoopMeshBuilder()
{
    free(triangles);
}