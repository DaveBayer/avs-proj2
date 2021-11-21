/**
 * @file    tree_mesh_builder.h
 *
 * @author  David Bayer <xbayer09@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    2021/11/16
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include <unordered_map>

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    uint decomposeOctree(Vec3_t<uint>, uint, const ParametricScalarField &);
    uint decomposeOctree(Vec3_t<float>, uint, const ParametricScalarField &);
    uint decomposeOctree(uint index, uint size, const ParametricScalarField &field);
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }
public:
    static const uint depth_limit;
private:
    std::unordered_map<uint, float> sphere_radius;
    std::vector<Triangle_t> mTriangles;
};

#endif // TREE_MESH_BUILDER_H
