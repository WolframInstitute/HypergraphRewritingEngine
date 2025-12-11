#pragma once

#include <cstdint>
#include <cstddef>

namespace viz::gal {

// Forward declarations
class Device;
class Buffer;
class Texture;
class Sampler;
class Shader;
class RenderPipeline;
class ComputePipeline;
class BindGroupLayout;
class BindGroup;
class CommandEncoder;
class CommandBuffer;
class RenderPassEncoder;
class ComputePassEncoder;
class Swapchain;
class Fence;
class Semaphore;

// Handle type for opaque backend resources
using Handle = void*;

// Backend selection
enum class Backend : uint8_t {
    Vulkan,
    WebGPU,
    Auto  // Select based on platform
};

// Pixel/texture formats
enum class Format : uint32_t {
    Undefined = 0,

    // 8-bit formats
    R8_UNORM,
    R8_SNORM,
    R8_UINT,
    R8_SINT,

    // 16-bit formats
    R16_UINT,
    R16_SINT,
    R16_FLOAT,
    RG8_UNORM,
    RG8_SNORM,
    RG8_UINT,
    RG8_SINT,

    // 32-bit formats
    R32_UINT,
    R32_SINT,
    R32_FLOAT,
    RG16_UINT,
    RG16_SINT,
    RG16_FLOAT,
    RGBA8_UNORM,
    RGBA8_SNORM,
    RGBA8_UINT,
    RGBA8_SINT,
    RGBA8_SRGB,
    BGRA8_UNORM,
    BGRA8_SRGB,

    // 64-bit formats
    RG32_UINT,
    RG32_SINT,
    RG32_FLOAT,
    RGB32_FLOAT,
    RGBA16_UINT,
    RGBA16_SINT,
    RGBA16_FLOAT,

    // 128-bit formats
    RGBA32_UINT,
    RGBA32_SINT,
    RGBA32_FLOAT,

    // Depth/stencil formats
    D16_UNORM,
    D24_UNORM_S8_UINT,
    D32_FLOAT,
    D32_FLOAT_S8_UINT,

    // Compressed formats (optional)
    BC1_RGBA_UNORM,
    BC1_RGBA_SRGB,
    BC3_RGBA_UNORM,
    BC3_RGBA_SRGB,
    BC7_RGBA_UNORM,
    BC7_RGBA_SRGB,
};

// Buffer usage flags (can be combined)
enum class BufferUsage : uint32_t {
    None            = 0,
    Vertex          = 1 << 0,
    Index           = 1 << 1,
    Uniform         = 1 << 2,
    Storage         = 1 << 3,
    Indirect        = 1 << 4,
    TransferSrc     = 1 << 5,
    TransferDst     = 1 << 6,
};

inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline BufferUsage operator&(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline bool has_flag(BufferUsage flags, BufferUsage flag) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

// Memory location hints
enum class MemoryLocation : uint8_t {
    GPU_ONLY,       // Device local, not host visible
    CPU_TO_GPU,     // Host visible, write-combined (staging, uniforms)
    GPU_TO_CPU,     // Host visible, cached (readback)
};

// Texture usage flags
enum class TextureUsage : uint32_t {
    None            = 0,
    Sampled         = 1 << 0,
    Storage         = 1 << 1,
    RenderTarget    = 1 << 2,
    DepthStencil    = 1 << 3,
    TransferSrc     = 1 << 4,
    TransferDst     = 1 << 5,
};

inline TextureUsage operator|(TextureUsage a, TextureUsage b) {
    return static_cast<TextureUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline TextureUsage operator&(TextureUsage a, TextureUsage b) {
    return static_cast<TextureUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline bool has_flag(TextureUsage flags, TextureUsage flag) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

// Texture dimensions
enum class TextureDimension : uint8_t {
    Tex1D,
    Tex2D,
    Tex3D,
    Cube,
};

// Primitive topology
enum class PrimitiveTopology : uint8_t {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
};

// Index format
enum class IndexFormat : uint8_t {
    Uint16,
    Uint32,
};

// Cull mode
enum class CullMode : uint8_t {
    None,
    Front,
    Back,
};

// Front face winding
enum class FrontFace : uint8_t {
    CCW,  // Counter-clockwise
    CW,   // Clockwise
};

// Polygon fill mode
enum class PolygonMode : uint8_t {
    Fill,
    Line,
    Point,
};

// Compare function (depth/stencil)
enum class CompareFunc : uint8_t {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
};

// Blend factor
enum class BlendFactor : uint8_t {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    SrcAlphaSaturate,
};

// Blend operation
enum class BlendOp : uint8_t {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
};

// Sampler filter
enum class Filter : uint8_t {
    Nearest,
    Linear,
};

// Sampler mipmap mode
enum class MipmapMode : uint8_t {
    Nearest,
    Linear,
};

// Sampler address mode
enum class AddressMode : uint8_t {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
};

// Shader stage flags (can be combined)
enum class ShaderStage : uint32_t {
    Vertex   = 1 << 0,
    Fragment = 1 << 1,
    Compute  = 1 << 2,
    All      = Vertex | Fragment | Compute,
};

inline ShaderStage operator|(ShaderStage a, ShaderStage b) {
    return static_cast<ShaderStage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline ShaderStage operator&(ShaderStage a, ShaderStage b) {
    return static_cast<ShaderStage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

// Vertex step mode (per-vertex or per-instance)
enum class VertexStepMode : uint8_t {
    Vertex,
    Instance,
};

// Load operation for render pass attachments
enum class LoadOp : uint8_t {
    Load,
    Clear,
    DontCare,
};

// Store operation for render pass attachments
enum class StoreOp : uint8_t {
    Store,
    DontCare,
};

// Binding types for bind group layouts
enum class BindingType : uint8_t {
    UniformBuffer,
    StorageBuffer,
    StorageBufferReadOnly,
    Sampler,
    SampledTexture,
    StorageTexture,
    CombinedTextureSampler,
};

// Color write mask
enum class ColorWriteMask : uint8_t {
    None  = 0,
    R     = 1 << 0,
    G     = 1 << 1,
    B     = 1 << 2,
    A     = 1 << 3,
    All   = R | G | B | A,
};

inline ColorWriteMask operator|(ColorWriteMask a, ColorWriteMask b) {
    return static_cast<ColorWriteMask>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

// Structures

struct Extent2D {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct Extent3D {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
};

struct Offset2D {
    int32_t x = 0;
    int32_t y = 0;
};

struct Offset3D {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
};

struct Viewport {
    float x = 0;
    float y = 0;
    float width = 0;
    float height = 0;
    float min_depth = 0;
    float max_depth = 1;
};

struct Scissor {
    int32_t x = 0;
    int32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ClearColor {
    float r = 0, g = 0, b = 0, a = 1;
};

struct ClearDepthStencil {
    float depth = 1.0f;
    uint32_t stencil = 0;
};

struct VertexAttribute {
    uint32_t shader_location = 0;
    Format format = Format::Undefined;
    uint32_t offset = 0;
};

struct VertexBufferLayout {
    uint32_t stride = 0;
    VertexStepMode step_mode = VertexStepMode::Vertex;
    const VertexAttribute* attributes = nullptr;
    uint32_t attribute_count = 0;
};

struct BlendState {
    bool blend_enable = false;
    BlendFactor src_color = BlendFactor::One;
    BlendFactor dst_color = BlendFactor::Zero;
    BlendOp color_op = BlendOp::Add;
    BlendFactor src_alpha = BlendFactor::One;
    BlendFactor dst_alpha = BlendFactor::Zero;
    BlendOp alpha_op = BlendOp::Add;
    uint32_t write_mask = 0xF;  // RGBA = 0b1111

    static BlendState alpha_blend() {
        BlendState s;
        s.blend_enable = true;
        s.src_color = BlendFactor::SrcAlpha;
        s.dst_color = BlendFactor::OneMinusSrcAlpha;
        s.src_alpha = BlendFactor::One;
        s.dst_alpha = BlendFactor::OneMinusSrcAlpha;
        return s;
    }

    static BlendState additive() {
        BlendState s;
        s.blend_enable = true;
        s.src_color = BlendFactor::One;
        s.dst_color = BlendFactor::One;
        s.src_alpha = BlendFactor::One;
        s.dst_alpha = BlendFactor::One;
        return s;
    }
};

struct DepthStencilState {
    bool depth_test_enable = true;
    bool depth_write_enable = true;
    CompareFunc depth_compare = CompareFunc::Less;
    bool stencil_test_enable = false;
    // Stencil ops omitted for brevity, can add later
};

struct RasterizerState {
    PolygonMode polygon_mode = PolygonMode::Fill;
    CullMode cull_mode = CullMode::Back;
    FrontFace front_face = FrontFace::CCW;
    bool depth_clamp_enable = false;
    bool depth_bias_enable = false;
    float depth_bias_constant = 0;
    float depth_bias_slope = 0;
    float depth_bias_clamp = 0;
    float line_width = 1.0f;
};

struct MultisampleState {
    uint32_t count = 1;
    bool sample_shading_enable = false;
    float min_sample_shading = 1.0f;
    uint32_t sample_mask = 0;
    bool alpha_to_coverage_enable = false;
};

struct BindGroupLayoutEntry {
    uint32_t binding = 0;
    ShaderStage visibility = ShaderStage::Vertex;  // Can be combined with |
    BindingType type = BindingType::UniformBuffer;
    uint32_t count = 1;  // For arrays
};

struct ColorAttachment {
    Texture* texture = nullptr;
    Texture* resolve_texture = nullptr;  // For MSAA resolve
    LoadOp load_op = LoadOp::Clear;
    StoreOp store_op = StoreOp::Store;
    ClearColor clear_value;
};

struct DepthStencilAttachment {
    Texture* texture = nullptr;
    LoadOp depth_load_op = LoadOp::Clear;
    StoreOp depth_store_op = StoreOp::Store;
    LoadOp stencil_load_op = LoadOp::DontCare;
    StoreOp stencil_store_op = StoreOp::DontCare;
    ClearDepthStencil clear_value;
};

// Utility functions
inline uint32_t format_size(Format format) {
    switch (format) {
        case Format::R8_UNORM:
        case Format::R8_SNORM:
        case Format::R8_UINT:
        case Format::R8_SINT:
            return 1;
        case Format::R16_UINT:
        case Format::R16_SINT:
        case Format::R16_FLOAT:
        case Format::RG8_UNORM:
        case Format::RG8_SNORM:
        case Format::RG8_UINT:
        case Format::RG8_SINT:
            return 2;
        case Format::R32_UINT:
        case Format::R32_SINT:
        case Format::R32_FLOAT:
        case Format::RG16_UINT:
        case Format::RG16_SINT:
        case Format::RG16_FLOAT:
        case Format::RGBA8_UNORM:
        case Format::RGBA8_SNORM:
        case Format::RGBA8_UINT:
        case Format::RGBA8_SINT:
        case Format::RGBA8_SRGB:
        case Format::BGRA8_UNORM:
        case Format::BGRA8_SRGB:
            return 4;
        case Format::RG32_UINT:
        case Format::RG32_SINT:
        case Format::RG32_FLOAT:
        case Format::RGBA16_UINT:
        case Format::RGBA16_SINT:
        case Format::RGBA16_FLOAT:
            return 8;
        case Format::RGB32_FLOAT:
            return 12;
        case Format::RGBA32_UINT:
        case Format::RGBA32_SINT:
        case Format::RGBA32_FLOAT:
            return 16;
        case Format::D16_UNORM:
            return 2;
        case Format::D24_UNORM_S8_UINT:
        case Format::D32_FLOAT:
            return 4;
        case Format::D32_FLOAT_S8_UINT:
            return 8;
        default:
            return 0;
    }
}

inline bool is_depth_format(Format format) {
    switch (format) {
        case Format::D16_UNORM:
        case Format::D24_UNORM_S8_UINT:
        case Format::D32_FLOAT:
        case Format::D32_FLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

inline bool has_stencil(Format format) {
    switch (format) {
        case Format::D24_UNORM_S8_UINT:
        case Format::D32_FLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

} // namespace viz::gal
