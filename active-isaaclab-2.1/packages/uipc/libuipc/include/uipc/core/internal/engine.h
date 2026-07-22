#pragma once
#include <string>
#include <uipc/common/dllexport.h>
#include <uipc/common/smart_pointer.h>
#include <uipc/core/i_engine.h>
#include <uipc/backend/visitors/world_visitor.h>
#include <uipc/common/exception.h>

namespace uipc::core::internal
{
class World;

class UIPC_CORE_API Engine final
{
    class Impl;

  public:
    Engine(std::string_view backend_name,
           std::string_view workspace = "./",
           const Json&      config    = default_config());
    ~Engine();

    std::string_view         backend_name() const noexcept;
    std::string_view         workspace() const noexcept;
    EngineStatusCollection&  status();
    const FeatureCollection& features();

    Json to_json() const;

    static Json default_config();

  private:
    friend class internal::World;
    // only be called by internal::world
    void  init(internal::World& world);
    void  advance();
    void  backward();
    void  sync();
    void  retrieve();
    bool  dump();
    bool  recover(SizeT dst_frame);
    bool  write_vertex_pos_to_sim(span<const Vector3> positions, IndexT global_vertex_offset, IndexT local_vertex_offset, SizeT vertex_count, string system_name);
    bool  write_global_vertex_pos_pair_to_sim(span<const Vector3> previous_positions, span<const Vector3> current_positions, IndexT global_vertex_offset, SizeT vertex_count);
    bool  write_kinematic_abd_pose_pair_to_sim(IndexT body_id, const Matrix4x4& previous_transform, const Matrix4x4& current_transform, Float dt);
    vector<Vector12> read_kinematic_abd_state_from_sim(IndexT body_id);
    SizeT frame() const;

    U<Impl> m_impl;
};
}  // namespace uipc::core::internal
