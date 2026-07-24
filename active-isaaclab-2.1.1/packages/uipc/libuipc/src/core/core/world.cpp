#include <uipc/core/world.h>
#include <uipc/core/engine.h>
#include <uipc/backend/visitors/world_visitor.h>
#include <uipc/core/sanity_checker.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/zip.h>
#include <uipc/common/uipc.h>
#include <dylib.hpp>
#include <uipc/backend/module_init_info.h>
#include <uipc/core/internal/world.h>
#include <iostream>

namespace uipc::core
{
World::World(Engine& e) noexcept
    : m_internal{uipc::make_shared<internal::World>(*e.m_internal)}
{
}

World::World(S<internal::World> w) noexcept
    : m_internal{std::move(w)}
{
}

World::~World() {}

void World::init(Scene& s)
{
    //TODO
    std::cout << "Init test " << "\n";
    m_internal->init(*s.m_internal);
}

void World::advance()
{
    m_internal->advance();
}

void World::sync()
{
    m_internal->sync();
}

void World::retrieve()
{
    m_internal->retrieve();
}

void World::backward()
{
    m_internal->backward();
}

bool World::dump()
{
    return m_internal->dump();
}

bool World::recover(SizeT aim_frame)
{
    return m_internal->recover(aim_frame);
}

bool World::write_vertex_pos_to_sim(span<const Vector3> positions, IndexT global_vertex_offset, IndexT local_vertex_offset, SizeT vertex_count, string system_name)
{   
    // std::cout << "write_vertex_pos test " << "\n";
    return m_internal->write_vertex_pos_to_sim(positions, global_vertex_offset, local_vertex_offset, vertex_count, system_name);
}

bool World::write_global_vertex_pos_pair_to_sim(span<const Vector3> previous_positions, span<const Vector3> current_positions, IndexT global_vertex_offset, SizeT vertex_count)
{
    return m_internal->write_global_vertex_pos_pair_to_sim(
        previous_positions, current_positions, global_vertex_offset, vertex_count);
}

bool World::write_kinematic_abd_pose_pair_to_sim(IndexT body_id, const Matrix4x4& previous_transform, const Matrix4x4& current_transform, Float dt)
{
    return m_internal->write_kinematic_abd_pose_pair_to_sim(
        body_id, previous_transform, current_transform, dt);
}

vector<Vector12> World::read_kinematic_abd_state_from_sim(IndexT body_id)
{
    return m_internal->read_kinematic_abd_state_from_sim(body_id);
}

bool World::is_valid() const
{
    return m_internal->is_valid();
}

SizeT World::frame() const
{
    return m_internal->frame();
}

const FeatureCollection& World::features() const
{
    return m_internal->features();
}
}  // namespace uipc::core
