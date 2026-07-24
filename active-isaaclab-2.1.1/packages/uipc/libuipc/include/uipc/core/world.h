#pragma once

#include <uipc/core/scene.h>
#include <uipc/core/feature_collection.h>

namespace uipc::backend
{
class WorldVisitor;
}

namespace uipc::core::internal
{
class World;
}

namespace uipc::core
{
class Engine;

class UIPC_CORE_API World final
{
    friend class backend::WorldVisitor;
    friend class SanityChecker;

  public:
    World(Engine& e) noexcept;
    ~World();

    World(const World&)            = delete;
    World(World&&)                 = default;
    World& operator=(const World&) = delete;
    World& operator=(World&&)      = default;

    void init(Scene& s);

    void advance();
    void sync();
    void retrieve();
    void backward();
    bool dump();
    bool recover(SizeT aim_frame = ~0ull);
    bool write_vertex_pos_to_sim(span<const Vector3> positions, IndexT global_vertex_offset, IndexT local_vertex_offset, SizeT vertex_count, string system_name);
    bool write_global_vertex_pos_pair_to_sim(span<const Vector3> previous_positions,
                                             span<const Vector3> current_positions,
                                             IndexT global_vertex_offset,
                                             SizeT vertex_count);
    bool write_kinematic_abd_pose_pair_to_sim(IndexT body_id,
                                               const Matrix4x4& previous_transform,
                                               const Matrix4x4& current_transform,
                                               Float dt);
    vector<Vector12> read_kinematic_abd_state_from_sim(IndexT body_id);
    bool is_valid() const;

    SizeT frame() const;

    const FeatureCollection& features() const;

  private:
    // Allow create a core::World from a core::internal::World
    World(S<internal::World> w) noexcept;
    S<internal::World> m_internal;
};
}  // namespace uipc::core
