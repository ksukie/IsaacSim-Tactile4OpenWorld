#pragma once
#include <cub/version.cuh>
#include <muda/launch/launch_base.h>
#include <muda/buffer.h>
#include <muda/container.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/compute_graph/compute_graph.h>
#include <muda/launch/stream.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
namespace cub
{
struct Equality
{
    template <typename A, typename B>
    __host__ __device__ constexpr bool operator()(const A& a, const B& b) const
    {
        return a == b;
    }
};

struct Sum
{
    template <typename A, typename B>
    __host__ __device__ constexpr auto operator()(const A& a, const B& b) const -> decltype(a + b)
    {
        return a + b;
    }
};
}  // namespace cub
#endif

namespace muda
{
template <typename Derive>
class CubWrapper : public LaunchBase<Derive>
{
  protected:
    std::byte* prepare_buffer(size_t reqSize)
    {
        return m_muda_stream->workspace(reqSize);
    }

  public:
    CubWrapper(Stream& stream = Stream::Default())
        : LaunchBase<Derive>(stream)
        , m_muda_stream(&stream)
    {
    }

    // meaningless for cub, so we just delete it
    void kernel_name(std::string_view) = delete;

    Stream* m_muda_stream = nullptr;
};
}  // namespace muda
