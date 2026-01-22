#include <platform/window.hpp>

#ifdef VIZ_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <windowsx.h>
#include <shellscalingapi.h>

#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace viz::platform {

// Enable Per-Monitor DPI awareness (Windows 8.1+)
// This ensures we get physical pixel coordinates, not virtualized/scaled ones
static void enable_dpi_awareness() {
    // Try SetProcessDpiAwarenessContext (Windows 10 1703+) first
    using SetProcessDpiAwarenessContextFunc = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (user32) {
        auto SetProcessDpiAwarenessContextPtr = reinterpret_cast<SetProcessDpiAwarenessContextFunc>(
            GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
        if (SetProcessDpiAwarenessContextPtr) {
            // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ((DPI_AWARENESS_CONTEXT)-4)
            if (SetProcessDpiAwarenessContextPtr(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
                return;
            }
        }
    }

    // Fallback: SetProcessDpiAwareness (Windows 8.1+)
    using SetProcessDpiAwarenessFunc = HRESULT(WINAPI*)(int);
    HMODULE shcore = LoadLibraryW(L"shcore.dll");
    if (shcore) {
        auto SetProcessDpiAwarenessPtr = reinterpret_cast<SetProcessDpiAwarenessFunc>(
            GetProcAddress(shcore, "SetProcessDpiAwareness"));
        if (SetProcessDpiAwarenessPtr) {
            // PROCESS_PER_MONITOR_DPI_AWARE = 2
            SetProcessDpiAwarenessPtr(2);
        }
        FreeLibrary(shcore);
    }
}

// Call DPI awareness setup before any windows are created
static bool dpi_initialized = (enable_dpi_awareness(), true);

// Windows virtual key to our KeyCode mapping
// lParam is needed to distinguish left/right modifier keys
static KeyCode vk_to_keycode(WPARAM vk, LPARAM lParam) {
    // Letters (VK codes for A-Z are same as ASCII 'A'-'Z')
    if (vk >= 'A' && vk <= 'Z') {
        return static_cast<KeyCode>(vk);
    }

    // Numbers
    if (vk >= '0' && vk <= '9') {
        return static_cast<KeyCode>(vk);
    }

    // Function keys
    if (vk >= VK_F1 && vk <= VK_F12) {
        return static_cast<KeyCode>(static_cast<uint16_t>(KeyCode::F1) + (vk - VK_F1));
    }

    // Handle generic modifier keys - Windows sends VK_SHIFT/VK_CONTROL/VK_MENU
    // and we need to use MapVirtualKey with the scan code to get left/right
    UINT scancode = (lParam >> 16) & 0xFF;
    bool extended = (lParam >> 24) & 0x1;

    switch (vk) {
        case VK_SHIFT: {
            // Use MapVirtualKey to distinguish left/right shift
            UINT mapped_vk = MapVirtualKey(scancode, MAPVK_VSC_TO_VK_EX);
            return (mapped_vk == VK_RSHIFT) ? KeyCode::RightShift : KeyCode::LeftShift;
        }
        case VK_CONTROL:
            return extended ? KeyCode::RightControl : KeyCode::LeftControl;
        case VK_MENU:
            return extended ? KeyCode::RightAlt : KeyCode::LeftAlt;
    }

    // Special keys
    switch (vk) {
        case VK_ESCAPE: return KeyCode::Escape;
        case VK_RETURN: return KeyCode::Enter;
        case VK_TAB: return KeyCode::Tab;
        case VK_BACK: return KeyCode::Backspace;
        case VK_INSERT: return KeyCode::Insert;
        case VK_DELETE: return KeyCode::Delete;
        case VK_RIGHT: return KeyCode::Right;
        case VK_LEFT: return KeyCode::Left;
        case VK_DOWN: return KeyCode::Down;
        case VK_UP: return KeyCode::Up;
        case VK_PRIOR: return KeyCode::PageUp;
        case VK_NEXT: return KeyCode::PageDown;
        case VK_HOME: return KeyCode::Home;
        case VK_END: return KeyCode::End;
        case VK_LSHIFT: return KeyCode::LeftShift;
        case VK_LCONTROL: return KeyCode::LeftControl;
        case VK_LMENU: return KeyCode::LeftAlt;
        case VK_LWIN: return KeyCode::LeftSuper;
        case VK_RSHIFT: return KeyCode::RightShift;
        case VK_RCONTROL: return KeyCode::RightControl;
        case VK_RMENU: return KeyCode::RightAlt;
        case VK_RWIN: return KeyCode::RightSuper;
        case VK_SPACE: return KeyCode::Space;
        case VK_OEM_MINUS: return KeyCode::Minus;
        case VK_OEM_PLUS: return KeyCode::Equal;
        case VK_OEM_4: return KeyCode::LeftBracket;
        case VK_OEM_6: return KeyCode::RightBracket;
        case VK_OEM_5: return KeyCode::Backslash;
        case VK_OEM_1: return KeyCode::Semicolon;
        case VK_OEM_7: return KeyCode::Apostrophe;
        case VK_OEM_3: return KeyCode::Grave;
        case VK_OEM_COMMA: return KeyCode::Comma;
        case VK_OEM_PERIOD: return KeyCode::Period;
        case VK_OEM_2: return KeyCode::Slash;
        case VK_CAPITAL: return KeyCode::CapsLock;
        case VK_SCROLL: return KeyCode::ScrollLock;
        case VK_NUMLOCK: return KeyCode::NumLock;
        case VK_SNAPSHOT: return KeyCode::PrintScreen;
        case VK_PAUSE: return KeyCode::Pause;
        default: return KeyCode::Unknown;
    }
}

static Modifiers get_current_modifiers() {
    Modifiers mods = Modifiers::None;
    if (GetKeyState(VK_SHIFT) & 0x8000) mods = mods | Modifiers::Shift;
    if (GetKeyState(VK_CONTROL) & 0x8000) mods = mods | Modifiers::Control;
    if (GetKeyState(VK_MENU) & 0x8000) mods = mods | Modifiers::Alt;
    if ((GetKeyState(VK_LWIN) | GetKeyState(VK_RWIN)) & 0x8000) mods = mods | Modifiers::Super;
    return mods;
}

class Win32Window : public Window {
public:
    Win32Window() = default;
    ~Win32Window() override { destroy(); }

    bool initialize(const WindowDesc& desc);
    void destroy();

    // Window interface
    bool is_open() const override { return open_; }
    bool is_minimized() const override { return minimized_; }
    bool is_focused() const override { return focused_; }

    uint32_t get_width() const override { return width_; }
    uint32_t get_height() const override { return height_; }
    void set_title(const char* title) override;
    void set_size(uint32_t width, uint32_t height) override;

    void poll_events() override;
    void wait_events() override;

    void set_callbacks(const WindowCallbacks& callbacks) override { callbacks_ = callbacks; }

    bool is_key_pressed(KeyCode key) const override;
    bool is_mouse_button_pressed(MouseButton button) const override;
    void get_mouse_position(int& x, int& y) const override;

    void request_close() override { open_ = false; }

    void focus() override {
        if (hwnd_) {
            // Bring window to front
            SetForegroundWindow(hwnd_);
            // Also set focus
            SetFocus(hwnd_);
        }
    }

    void* get_native_display() const override { return hinstance_; }
    void* get_native_window() const override { return hwnd_; }

    // Static window procedure
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

private:
    LRESULT handle_message(UINT msg, WPARAM wParam, LPARAM lParam);
    void update_key_state(KeyCode key, bool pressed);

    HINSTANCE hinstance_ = nullptr;
    HWND hwnd_ = nullptr;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    int mouse_x_ = 0;
    int mouse_y_ = 0;

    bool open_ = false;
    bool minimized_ = false;
    bool focused_ = true;

    // Input state
    bool key_states_[512] = {false};
    bool mouse_button_states_[8] = {false};

    WindowCallbacks callbacks_;

    // Class name for window registration
    static constexpr const char* WINDOW_CLASS_NAME = "VizWindowClass";
    static bool class_registered_;
};

bool Win32Window::class_registered_ = false;

bool Win32Window::initialize(const WindowDesc& desc) {
    hinstance_ = GetModuleHandle(nullptr);

    // Register window class (once)
    if (!class_registered_) {
        WNDCLASSEXA wc = {};
        wc.cbSize = sizeof(WNDCLASSEXA);
        wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = hinstance_;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
        wc.lpszClassName = WINDOW_CLASS_NAME;

        if (!RegisterClassExA(&wc)) {
            std::cerr << "Failed to register window class" << std::endl;
            return false;
        }
        class_registered_ = true;
    }

    // Calculate window size with decorations
    DWORD style = WS_OVERLAPPEDWINDOW;
    if (!desc.resizable) {
        style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
    }
    if (!desc.decorated) {
        style = WS_POPUP;
    }

    RECT rect = {0, 0, static_cast<LONG>(desc.width), static_cast<LONG>(desc.height)};
    AdjustWindowRect(&rect, style, FALSE);

    int window_width = rect.right - rect.left;
    int window_height = rect.bottom - rect.top;

    // Create window using ANSI API (title is already char*)
    hwnd_ = CreateWindowExA(
        0,
        WINDOW_CLASS_NAME,
        desc.title,
        style,
        CW_USEDEFAULT, CW_USEDEFAULT,
        window_width, window_height,
        nullptr,
        nullptr,
        hinstance_,
        this  // Pass this pointer for WM_CREATE
    );

    if (!hwnd_) {
        std::cerr << "Failed to create window" << std::endl;
        return false;
    }

    // Get actual client area size (in physical pixels with DPI awareness)
    // This is what Vulkan swapchain will use
    RECT client_rect;
    GetClientRect(hwnd_, &client_rect);
    width_ = static_cast<uint32_t>(client_rect.right - client_rect.left);
    height_ = static_cast<uint32_t>(client_rect.bottom - client_rect.top);

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);

    open_ = true;
    return true;
}

void Win32Window::destroy() {
    if (hwnd_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
    open_ = false;
}

void Win32Window::set_title(const char* title) {
    if (!hwnd_) return;
    SetWindowTextA(hwnd_, title);
}

void Win32Window::set_size(uint32_t width, uint32_t height) {
    if (!hwnd_) return;

    DWORD style = static_cast<DWORD>(GetWindowLongPtr(hwnd_, GWL_STYLE));
    RECT rect = {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, style, FALSE);

    SetWindowPos(hwnd_, nullptr, 0, 0,
                 rect.right - rect.left, rect.bottom - rect.top,
                 SWP_NOMOVE | SWP_NOZORDER);
}

void Win32Window::poll_events() {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            open_ = false;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

void Win32Window::wait_events() {
    MSG msg;
    if (GetMessage(&msg, nullptr, 0, 0)) {
        if (msg.message == WM_QUIT) {
            open_ = false;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Also process any other pending events
    poll_events();
}

LRESULT CALLBACK Win32Window::WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Win32Window* window = nullptr;

    if (msg == WM_CREATE) {
        auto* create = reinterpret_cast<CREATESTRUCT*>(lParam);
        window = static_cast<Win32Window*>(create->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window));
    } else {
        window = reinterpret_cast<Win32Window*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }

    if (window) {
        return window->handle_message(msg, wParam, lParam);
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

LRESULT Win32Window::handle_message(UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_SIZE: {
            uint32_t new_width = LOWORD(lParam);
            uint32_t new_height = HIWORD(lParam);

            if (wParam == SIZE_MINIMIZED) {
                minimized_ = true;
            } else {
                minimized_ = false;
                if (new_width != width_ || new_height != height_) {
                    width_ = new_width;
                    height_ = new_height;
                    if (callbacks_.on_resize) {
                        callbacks_.on_resize(width_, height_);
                    }
                }
            }
            return 0;
        }

        case WM_CLOSE:
            open_ = false;
            if (callbacks_.on_close) {
                callbacks_.on_close();
            }
            DestroyWindow(hwnd_);
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_KEYDOWN:
        case WM_SYSKEYDOWN: {
            KeyCode key = vk_to_keycode(wParam, lParam);
            Modifiers mods = get_current_modifiers();
            update_key_state(key, true);

            if (callbacks_.on_key) {
                callbacks_.on_key(key, true, mods);
            }

            // Let Alt+F4 through to DefWindowProc so it generates WM_CLOSE
            if (msg == WM_SYSKEYDOWN && wParam == VK_F4 && (GetKeyState(VK_MENU) & 0x8000)) {
                return DefWindowProc(hwnd_, msg, wParam, lParam);
            }
            return 0;
        }

        case WM_KEYUP:
        case WM_SYSKEYUP: {
            KeyCode key = vk_to_keycode(wParam, lParam);
            Modifiers mods = get_current_modifiers();
            update_key_state(key, false);

            if (callbacks_.on_key) {
                callbacks_.on_key(key, false, mods);
            }
            return 0;
        }

        case WM_CHAR: {
            if (callbacks_.on_char && wParam >= 0x20 && wParam < 0x10000) {
                callbacks_.on_char(static_cast<uint32_t>(wParam));
            }
            return 0;
        }

        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP: {
            bool pressed = (msg == WM_LBUTTONDOWN);
            mouse_button_states_[static_cast<uint8_t>(MouseButton::Left)] = pressed;

            if (pressed) SetCapture(hwnd_);
            else ReleaseCapture();

            if (callbacks_.on_mouse_button) {
                callbacks_.on_mouse_button(
                    MouseButton::Left, pressed,
                    GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam),
                    get_current_modifiers()
                );
            }
            return 0;
        }

        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP: {
            bool pressed = (msg == WM_RBUTTONDOWN);
            mouse_button_states_[static_cast<uint8_t>(MouseButton::Right)] = pressed;

            if (callbacks_.on_mouse_button) {
                callbacks_.on_mouse_button(
                    MouseButton::Right, pressed,
                    GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam),
                    get_current_modifiers()
                );
            }
            return 0;
        }

        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP: {
            bool pressed = (msg == WM_MBUTTONDOWN);
            mouse_button_states_[static_cast<uint8_t>(MouseButton::Middle)] = pressed;

            if (callbacks_.on_mouse_button) {
                callbacks_.on_mouse_button(
                    MouseButton::Middle, pressed,
                    GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam),
                    get_current_modifiers()
                );
            }
            return 0;
        }

        case WM_MOUSEMOVE: {
            mouse_x_ = GET_X_LPARAM(lParam);
            mouse_y_ = GET_Y_LPARAM(lParam);

            if (callbacks_.on_mouse_move) {
                callbacks_.on_mouse_move(mouse_x_, mouse_y_);
            }
            return 0;
        }

        case WM_MOUSEWHEEL: {
            if (callbacks_.on_scroll) {
                float delta = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
                callbacks_.on_scroll(0, delta);
            }
            return 0;
        }

        case WM_MOUSEHWHEEL: {
            if (callbacks_.on_scroll) {
                float delta = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
                callbacks_.on_scroll(delta, 0);
            }
            return 0;
        }

        case WM_SETFOCUS:
            focused_ = true;
            if (callbacks_.on_focus) {
                callbacks_.on_focus(true);
            }
            return 0;

        case WM_KILLFOCUS:
            focused_ = false;
            if (callbacks_.on_focus) {
                callbacks_.on_focus(false);
            }
            return 0;

        default:
            break;
    }

    return DefWindowProc(hwnd_, msg, wParam, lParam);
}

void Win32Window::update_key_state(KeyCode key, bool pressed) {
    uint16_t index = static_cast<uint16_t>(key);
    if (index < 512) {
        key_states_[index] = pressed;
    }
}

bool Win32Window::is_key_pressed(KeyCode key) const {
    uint16_t index = static_cast<uint16_t>(key);
    if (index < 512) {
        return key_states_[index];
    }
    return false;
}

bool Win32Window::is_mouse_button_pressed(MouseButton button) const {
    uint8_t index = static_cast<uint8_t>(button);
    if (index < 8) {
        return mouse_button_states_[index];
    }
    return false;
}

void Win32Window::get_mouse_position(int& x, int& y) const {
    x = mouse_x_;
    y = mouse_y_;
}

// Factory implementation
std::unique_ptr<Window> Window::create(const WindowDesc& desc) {
    auto window = std::make_unique<Win32Window>();
    if (window->initialize(desc)) {
        return window;
    }
    return nullptr;
}

} // namespace viz::platform

#endif // VIZ_PLATFORM_WINDOWS
