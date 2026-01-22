#pragma once

#include <cstdint>
#include <memory>
#include <functional>
#include <string>

namespace viz::platform {

// Forward declaration for Vulkan surface creation
namespace gal { class Device; }

// Input event types
enum class KeyCode : uint16_t {
    Unknown = 0,

    // Letters
    A = 'A', B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,

    // Numbers
    Num0 = '0', Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,

    // Function keys
    F1 = 256, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

    // Navigation
    Escape, Enter, Tab, Backspace, Insert, Delete,
    Right, Left, Down, Up, PageUp, PageDown, Home, End,

    // Modifiers
    LeftShift, LeftControl, LeftAlt, LeftSuper,
    RightShift, RightControl, RightAlt, RightSuper,

    // Other
    Space, Minus, Equal, LeftBracket, RightBracket,
    Backslash, Semicolon, Apostrophe, Grave, Comma, Period, Slash,
    CapsLock, ScrollLock, NumLock, PrintScreen, Pause,
};

enum class MouseButton : uint8_t {
    Left = 0,
    Right = 1,
    Middle = 2,
    Button4 = 3,
    Button5 = 4,
};

// Modifier key flags
enum class Modifiers : uint8_t {
    None = 0,
    Shift = 1 << 0,
    Control = 1 << 1,
    Alt = 1 << 2,
    Super = 1 << 3,
};

inline Modifiers operator|(Modifiers a, Modifiers b) {
    return static_cast<Modifiers>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
inline bool has_modifier(Modifiers mods, Modifiers mod) {
    return (static_cast<uint8_t>(mods) & static_cast<uint8_t>(mod)) != 0;
}

// Event callbacks
struct WindowCallbacks {
    std::function<void(uint32_t width, uint32_t height)> on_resize;
    std::function<void()> on_close;
    std::function<void(KeyCode key, bool pressed, Modifiers mods)> on_key;
    std::function<void(uint32_t codepoint)> on_char;
    std::function<void(MouseButton button, bool pressed, int x, int y, Modifiers mods)> on_mouse_button;
    std::function<void(int x, int y)> on_mouse_move;
    std::function<void(float dx, float dy)> on_scroll;
    std::function<void(bool focused)> on_focus;
};

// Window creation descriptor
struct WindowDesc {
    const char* title = "Hypergraph Visualization";
    uint32_t width = 1280;
    uint32_t height = 720;
    bool resizable = true;
    bool decorated = true;  // Window decorations (title bar, etc.)
    bool fullscreen = false;
};

// Abstract window interface
class Window {
public:
    virtual ~Window() = default;

    // Factory method
    static std::unique_ptr<Window> create(const WindowDesc& desc);

    // Window state
    virtual bool is_open() const = 0;
    virtual bool is_minimized() const = 0;
    virtual bool is_focused() const = 0;

    // Window properties
    virtual uint32_t get_width() const = 0;
    virtual uint32_t get_height() const = 0;
    virtual void set_title(const char* title) = 0;
    virtual void set_size(uint32_t width, uint32_t height) = 0;

    // Event processing
    virtual void poll_events() = 0;
    virtual void wait_events() = 0;  // Block until events arrive

    // Callbacks
    virtual void set_callbacks(const WindowCallbacks& callbacks) = 0;

    // Input state queries
    virtual bool is_key_pressed(KeyCode key) const = 0;
    virtual bool is_mouse_button_pressed(MouseButton button) const = 0;
    virtual void get_mouse_position(int& x, int& y) const = 0;

    // Close request
    virtual void request_close() = 0;

    // Bring window to front and give it focus
    virtual void focus() = 0;

    // Platform-specific handles for Vulkan surface creation
    // Returns platform-specific data needed to create VkSurface
    virtual void* get_native_display() const = 0;  // Display*/xcb_connection_t*/HINSTANCE
    virtual void* get_native_window() const = 0;   // Window/xcb_window_t/HWND
};

} // namespace viz::platform
