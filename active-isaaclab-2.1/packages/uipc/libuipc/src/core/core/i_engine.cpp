#include <uipc/core/i_engine.h>
#include <dylib.hpp>

namespace uipc::core
{
void IEngine::init(internal::World& w)
{
    do_init(w);
}

void IEngine::advance()
{
    do_advance();
}

void IEngine::backward()
{
    do_backward();
}

void IEngine::sync()
{
    do_sync();
}

void IEngine::retrieve()
{
    do_retrieve();
}

Json IEngine::to_json() const
{
    return do_to_json();
}

bool IEngine::dump()
{
    return do_dump();
}

bool IEngine::recover(SizeT dst_frame)
{
    return do_recover(dst_frame);
}

bool IEngine::write_vertex_pos_to_sim(span<const Vector3> positions, IndexT global_vertex_offset, IndexT local_vertex_offset, SizeT vertex_count, string system_name)
{
    return do_write_vertex_pos_to_sim(positions, global_vertex_offset, local_vertex_offset, vertex_count, system_name);
}

bool IEngine::write_global_vertex_pos_pair_to_sim(span<const Vector3> previous_positions, span<const Vector3> current_positions, IndexT global_vertex_offset, SizeT vertex_count)
{
    return do_write_global_vertex_pos_pair_to_sim(
        previous_positions, current_positions, global_vertex_offset, vertex_count);
}

bool IEngine::write_kinematic_abd_pose_pair_to_sim(IndexT body_id, const Matrix4x4& previous_transform, const Matrix4x4& current_transform, Float dt)
{
    return do_write_kinematic_abd_pose_pair_to_sim(body_id, previous_transform, current_transform, dt);
}

vector<Vector12> IEngine::read_kinematic_abd_state_from_sim(IndexT body_id)
{
    return do_read_kinematic_abd_state_from_sim(body_id);
}


SizeT IEngine::frame() const
{
    return get_frame();
}

EngineStatusCollection& IEngine::status()
{
    return get_status();
}

const FeatureCollection& IEngine::features() const
{
    return get_features();
}

Json IEngine::do_to_json() const
{
    return Json{};
}

bool IEngine::do_dump()
{
    return true;
}

bool IEngine::do_recover(SizeT dst_frame)
{
    return true;
}

bool IEngine::do_write_vertex_pos_to_sim(span<const Vector3> positions, IndexT global_vertex_offset, IndexT local_vertex_offset, SizeT vertex_count, string system_name)
{
    return true;
}

bool IEngine::do_write_global_vertex_pos_pair_to_sim(span<const Vector3>, span<const Vector3>, IndexT, SizeT)
{
    return false;
}

bool IEngine::do_write_kinematic_abd_pose_pair_to_sim(IndexT, const Matrix4x4&, const Matrix4x4&, Float)
{
    return false;
}

vector<Vector12> IEngine::do_read_kinematic_abd_state_from_sim(IndexT)
{
    return {};
}

}  // namespace uipc::core
