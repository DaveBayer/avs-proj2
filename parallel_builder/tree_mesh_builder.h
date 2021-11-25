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

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);
    ~TreeMeshBuilder();

protected:
    uint decomposeOctree(Vec3_t<uint>, uint, const ParametricScalarField &);
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return triangles; }
public:
    static const size_t init_malloc_amount;
    static const size_t realloc_amount;

    static const uint depth_limit;
    
private:
    size_t size;
    size_t capacity;
    Triangle_t *triangles;
    

public:
    class GridSizeException : public std::exception
    {
    public:
        GridSizeException(const std::string &message = "")
        : m_message("GridSize must be power of 2.\t" + message) {}
        const char *what() const noexcept { return m_message.c_str(); }

    private:
        std::string m_message;
    };
};

#endif // TREE_MESH_BUILDER_H
