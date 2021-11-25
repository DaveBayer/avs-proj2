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

const size_t TreeMeshBuilder::init_malloc_amount = 100000UL;
const size_t TreeMeshBuilder::realloc_amount = 20000UL;
const uint TreeMeshBuilder::depth_limit = 4U;

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
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

uint TreeMeshBuilder::decomposeOctree(Vec3_t<uint> pos, uint size, const ParametricScalarField &field)
{
    auto sphere_radius = [this](uint size) -> float
    {
        return mIsoLevel + sqrt(3.0) / 2.0 * static_cast<float>(size) * mGridResolution;
    };

    auto decompose = [](Vec3_t<uint> p, uint size) -> std::array<Vec3_t<uint>, 8UL>
    {
        uint sh = size >> 1;   //  (size / 2)
        return {{
            { p.x, p.y, p.z },              { p.x + sh, p.y, p.z },
            { p.x, p.y + sh, p.z },         { p.x, p.y, p.z + sh },
            { p.x + sh, p.y + sh, p.z },    { p.x, p.y + sh, p.z + sh },
            { p.x + sh, p.y, p.z + sh },    { p.x + sh, p.y + sh, p.z + sh }
        }};
    };

    auto cube_center = [this](Vec3_t<uint> p, uint size) -> Vec3_t<float>
    {
        uint sh = size >> 1;  //  size / 2
        return { (p.x + sh) * mGridResolution, (p.y + sh) * mGridResolution, (p.z + sh) * mGridResolution };
    };

    uint totalTriangles = 0U;
    
    if (size > depth_limit) {
        float r = sphere_radius(size);
        uint subcube_size = size >> 1;

        for (auto sc : decompose(pos, size)) {
            Vec3_t<float> sc_center_pt = cube_center(sc, subcube_size);

            if (!(evaluateFieldAt(sc_center_pt, field) > r)) {
#               pragma omp task shared(totalTriangles) firstprivate(subcube_size, field)
                totalTriangles += decomposeOctree(sc, subcube_size, field);
            }
        }

#       pragma omp taskwait

    } else {
        for (uint i = 0U; i < depth_limit; i++) {
            for (uint j = 0U; j < depth_limit; j++) {
                for (uint k = 0U; k < depth_limit; k++) {
                    totalTriangles += buildCube(Vec3_t<float>(pos.x + i, pos.y + j, pos.z + k), field);
                }
            }
        }
    }

    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    uint totalTriangles;

#   pragma omp parallel
#   pragma omp master
    totalTriangles = decomposeOctree(Vec3_t<uint>(0U, 0U, 0U), mGridSize, field);

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

TreeMeshBuilder::~TreeMeshBuilder()
{
    free(triangles);
}
