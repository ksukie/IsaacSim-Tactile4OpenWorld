#include "uipc/common/type_define.h"
#include <pyuipc/core/world.h>
#include <uipc/core/world.h>
#include <uipc/core/engine.h>
#include <pyuipc/as_numpy.h>

namespace pyuipc::core
{
using namespace uipc::core;

PyWorld::PyWorld(py::module& m)
{
    auto class_World = py::class_<World>(m, "World");

    class_World.def(py::init<Engine&>())
        .def("init", &World::init, py::arg("scene"))
        .def("advance", &World::advance)
        .def("sync", &World::sync)
        .def("retrieve", &World::retrieve)
        .def("dump", &World::dump)
        .def("recover", &World::recover, py::arg("dst_frame") = ~0ull)
        .def("backward", &World::backward)
        .def("frame", &World::frame)
        .def("features", &World::features, py::return_value_policy::reference_internal)
        .def("is_valid", &World::is_valid)
        .def("write_vertex_pos_to_sim", 
            [](World& self, py::array_t<Float> positions, IndexT global_vertex_offset, IndexT local_vertex_offset, SizeT vertex_count, string system_name)
            {return self.write_vertex_pos_to_sim(as_span_of<Vector3>(positions), IndexT(global_vertex_offset), IndexT(local_vertex_offset), SizeT(vertex_count), system_name); },
            py::arg("positions"),
            py::arg("global_vertex_offset"),
            py::arg("local_vertex_offset"),
            py::arg("vertex_count"),
            py::arg("system_name") = string{""}
        )
        .def("write_global_vertex_pos_pair_to_sim",
            [](World& self,
               py::array_t<Float, py::array::c_style | py::array::forcecast> previous_positions,
               py::array_t<Float, py::array::c_style | py::array::forcecast> current_positions,
               IndexT global_vertex_offset,
               SizeT vertex_count)
            {
                PYUIPC_ASSERT(previous_positions.ndim() == 2
                                  && previous_positions.shape(1) == 3,
                              "previous_positions must have shape (N,3)");
                PYUIPC_ASSERT(current_positions.ndim() == 2
                                  && current_positions.shape(1) == 3,
                              "current_positions must have shape (N,3)");
                PYUIPC_ASSERT(previous_positions.shape(0)
                                  == static_cast<py::ssize_t>(vertex_count)
                                  && current_positions.shape(0)
                                         == static_cast<py::ssize_t>(vertex_count),
                              "position arrays must contain vertex_count rows");
                return self.write_global_vertex_pos_pair_to_sim(
                    as_span_of<Vector3>(previous_positions),
                    as_span_of<Vector3>(current_positions),
                    global_vertex_offset,
                    vertex_count);
            },
            py::arg("previous_positions"),
            py::arg("current_positions"),
            py::arg("global_vertex_offset"),
            py::arg("vertex_count")
        )
        .def("write_kinematic_abd_pose_pair_to_sim",
            [](World& self,
               IndexT body_id,
               py::array_t<Float, py::array::c_style | py::array::forcecast> previous_transform,
               py::array_t<Float, py::array::c_style | py::array::forcecast> current_transform,
               Float dt)
            {
                PYUIPC_ASSERT(previous_transform.ndim() == 2
                                  && previous_transform.shape(0) == 4
                                  && previous_transform.shape(1) == 4,
                              "previous_transform must have shape (4,4)");
                PYUIPC_ASSERT(current_transform.ndim() == 2
                                  && current_transform.shape(0) == 4
                                  && current_transform.shape(1) == 4,
                              "current_transform must have shape (4,4)");
                Matrix4x4 previous;
                Matrix4x4 current;
                for(IndexT row = 0; row < 4; ++row)
                    for(IndexT column = 0; column < 4; ++column)
                    {
                        previous(row, column) = *previous_transform.data(row, column);
                        current(row, column) = *current_transform.data(row, column);
                    }
                return self.write_kinematic_abd_pose_pair_to_sim(
                    body_id, previous, current, dt);
            },
            py::arg("body_id"),
            py::arg("previous_transform"),
            py::arg("current_transform"),
            py::arg("dt")
        )
        .def("read_kinematic_abd_state_from_sim",
            [](World& self, IndexT body_id)
            {
                const vector<Vector12> state = self.read_kinematic_abd_state_from_sim(body_id);
                py::array_t<Float> result({static_cast<py::ssize_t>(state.size()), py::ssize_t{12}});
                auto output = result.mutable_unchecked<2>();
                for(py::ssize_t row = 0; row < static_cast<py::ssize_t>(state.size()); ++row)
                    for(py::ssize_t column = 0; column < 12; ++column)
                        output(row, column) = state[row](column);
                return result;
            },
            py::arg("body_id")
        );
}

}  // namespace pyuipc::core
