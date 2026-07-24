#include <typeinfo>
#include <backends/common/i_sim_system.h>
#include <backends/common/module.h>
#include <filesystem>
#include <backends/common/backend_path_tool.h>

namespace uipc::backend
{
void ISimSystem::build()
{
    //spdlog::info("Building system: {}", name());
    do_build();
}

void ISimSystem::make_engine_aware()
{
    set_engine_aware();
}

void ISimSystem::invalidate() noexcept
{
    set_invalid();
}

bool ISimSystem::is_valid() const noexcept
{
    return get_valid();
}

bool ISimSystem::is_building() const noexcept
{
    return get_is_building();
}

span<ISimSystem* const> ISimSystem::strong_dependencies() const noexcept
{
    return get_strong_dependencies();
}

span<ISimSystem* const> ISimSystem::weak_dependencies() const noexcept
{
    return get_weak_dependencies();
}

std::string_view ISimSystem::name() const noexcept
{
    return get_name();
}

bool ISimSystem::is_engine_aware() const noexcept
{
    return get_engine_aware();
}

Json ISimSystem::to_json() const
{
    return do_to_json();
}

bool ISimSystem::dump(DumpInfo& info)
{
    return do_dump(info);
}

bool ISimSystem::try_recover(RecoverInfo& info)
{
    return do_try_recover(info);
}

void ISimSystem::apply_recover(RecoverInfo& info)
{
    do_apply_recover(info);
}

void ISimSystem::clear_recover(RecoverInfo& info)
{
    do_clear_recover(info);
}

bool ISimSystem::write_vertex_pos_to_sim(span<const Vector3> positions, IndexT vertex_offset, SizeT vertex_count)
{
    return do_write_vertex_pos_to_sim(positions, vertex_offset, vertex_count);
}

bool ISimSystem::write_vertex_pos_pair_to_sim(span<const Vector3> previous_positions, span<const Vector3> current_positions, IndexT vertex_offset, SizeT vertex_count)
{
    return do_write_vertex_pos_pair_to_sim(
        previous_positions, current_positions, vertex_offset, vertex_count);
}

bool ISimSystem::write_kinematic_abd_pose_pair_to_sim(IndexT body_id, const Matrix4x4& previous_transform, const Matrix4x4& current_transform, Float dt)
{
    return do_write_kinematic_abd_pose_pair_to_sim(body_id, previous_transform, current_transform, dt);
}

vector<Vector12> ISimSystem::read_kinematic_abd_state_from_sim(IndexT body_id)
{
    return do_read_kinematic_abd_state_from_sim(body_id);
}

bool ISimSystem::do_dump(DumpInfo&)
{
    return true;
}

bool ISimSystem::do_try_recover(RecoverInfo&)
{
    return true;
}

bool ISimSystem::do_write_vertex_pos_to_sim(span<const Vector3> positions, IndexT vertex_offset, SizeT vertex_count)
{
    return true;
}

bool ISimSystem::do_write_vertex_pos_pair_to_sim(span<const Vector3>, span<const Vector3>, IndexT, SizeT)
{
    return false;
}

bool ISimSystem::do_write_kinematic_abd_pose_pair_to_sim(IndexT, const Matrix4x4&, const Matrix4x4&, Float)
{
    return false;
}

vector<Vector12> ISimSystem::do_read_kinematic_abd_state_from_sim(IndexT)
{
    return {};
}

void ISimSystem::do_apply_recover(RecoverInfo&) {}

void ISimSystem::do_clear_recover(RecoverInfo&) {}

ISimSystem::BaseInfo::BaseInfo(SizeT frame, std::string_view workspace, const Json& config) noexcept
    : m_frame(frame)
    , m_config(config)
    , m_workspace(workspace)
{
}

std::string_view ISimSystem::BaseInfo::workspace() const noexcept
{
    return m_workspace;
}

std::string ISimSystem::BaseInfo::dump_path(std::string_view _file_) const noexcept
{
    BackendPathTool tool{m_workspace};
    return tool.workspace(_file_, "dump").string();
}

const Json& ISimSystem::BaseInfo::config() const noexcept
{
    return m_config;
}

SizeT ISimSystem::BaseInfo::frame() const noexcept
{
    return m_frame;
}
}  // namespace uipc::backend
