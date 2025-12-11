#include <platform/window.hpp>

#ifdef VIZ_PLATFORM_LINUX

#include <xcb/xcb.h>
#include <xcb/xcb_keysyms.h>
#include <xcb/xcb_icccm.h>

#include <cstring>
#include <iostream>
#include <unordered_map>

namespace viz::platform {

// XCB keycode to our KeyCode mapping
static KeyCode xcb_keycode_to_keycode(xcb_keysym_t keysym) {
    // Letters (lowercase and uppercase)
    if (keysym >= 'a' && keysym <= 'z') {
        return static_cast<KeyCode>('A' + (keysym - 'a'));
    }
    if (keysym >= 'A' && keysym <= 'Z') {
        return static_cast<KeyCode>(keysym);
    }

    // Numbers
    if (keysym >= '0' && keysym <= '9') {
        return static_cast<KeyCode>(keysym);
    }

    // Function keys
    if (keysym >= 0xFFBE && keysym <= 0xFFC9) {  // F1-F12
        return static_cast<KeyCode>(static_cast<uint16_t>(KeyCode::F1) + (keysym - 0xFFBE));
    }

    // Special keys
    switch (keysym) {
        case 0xFF1B: return KeyCode::Escape;
        case 0xFF0D: return KeyCode::Enter;
        case 0xFF09: return KeyCode::Tab;
        case 0xFF08: return KeyCode::Backspace;
        case 0xFF63: return KeyCode::Insert;
        case 0xFFFF: return KeyCode::Delete;
        case 0xFF53: return KeyCode::Right;
        case 0xFF51: return KeyCode::Left;
        case 0xFF54: return KeyCode::Down;
        case 0xFF52: return KeyCode::Up;
        case 0xFF55: return KeyCode::PageUp;
        case 0xFF56: return KeyCode::PageDown;
        case 0xFF50: return KeyCode::Home;
        case 0xFF57: return KeyCode::End;
        case 0xFFE1: return KeyCode::LeftShift;
        case 0xFFE3: return KeyCode::LeftControl;
        case 0xFFE9: return KeyCode::LeftAlt;
        case 0xFFEB: return KeyCode::LeftSuper;
        case 0xFFE2: return KeyCode::RightShift;
        case 0xFFE4: return KeyCode::RightControl;
        case 0xFFEA: return KeyCode::RightAlt;
        case 0xFFEC: return KeyCode::RightSuper;
        case 0x0020: return KeyCode::Space;
        case 0x002D: return KeyCode::Minus;
        case 0x003D: return KeyCode::Equal;
        case 0x005B: return KeyCode::LeftBracket;
        case 0x005D: return KeyCode::RightBracket;
        case 0x005C: return KeyCode::Backslash;
        case 0x003B: return KeyCode::Semicolon;
        case 0x0027: return KeyCode::Apostrophe;
        case 0x0060: return KeyCode::Grave;
        case 0x002C: return KeyCode::Comma;
        case 0x002E: return KeyCode::Period;
        case 0x002F: return KeyCode::Slash;
        default: return KeyCode::Unknown;
    }
}

static Modifiers xcb_state_to_modifiers(uint16_t state) {
    Modifiers mods = Modifiers::None;
    if (state & XCB_MOD_MASK_SHIFT) mods = mods | Modifiers::Shift;
    if (state & XCB_MOD_MASK_CONTROL) mods = mods | Modifiers::Control;
    if (state & XCB_MOD_MASK_1) mods = mods | Modifiers::Alt;  // Mod1 is usually Alt
    if (state & XCB_MOD_MASK_4) mods = mods | Modifiers::Super;  // Mod4 is usually Super
    return mods;
}

class XcbWindow : public Window {
public:
    XcbWindow() = default;
    ~XcbWindow() override { destroy(); }

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

    void* get_native_display() const override { return connection_; }
    void* get_native_window() const override { return reinterpret_cast<void*>(static_cast<uintptr_t>(window_)); }

private:
    void handle_event(xcb_generic_event_t* event);
    void update_key_state(KeyCode key, bool pressed);

    xcb_connection_t* connection_ = nullptr;
    xcb_screen_t* screen_ = nullptr;
    xcb_window_t window_ = 0;
    xcb_key_symbols_t* key_symbols_ = nullptr;
    xcb_atom_t wm_protocols_atom_ = 0;
    xcb_atom_t wm_delete_window_atom_ = 0;

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
};

bool XcbWindow::initialize(const WindowDesc& desc) {
    // Connect to X server
    int screen_num;
    connection_ = xcb_connect(nullptr, &screen_num);

    if (xcb_connection_has_error(connection_)) {
        std::cerr << "Failed to connect to X server" << std::endl;
        return false;
    }

    // Get the screen
    const xcb_setup_t* setup = xcb_get_setup(connection_);
    xcb_screen_iterator_t iter = xcb_setup_roots_iterator(setup);
    for (int i = 0; i < screen_num; ++i) {
        xcb_screen_next(&iter);
    }
    screen_ = iter.data;

    // Create the window
    window_ = xcb_generate_id(connection_);

    uint32_t event_mask =
        XCB_EVENT_MASK_EXPOSURE |
        XCB_EVENT_MASK_KEY_PRESS |
        XCB_EVENT_MASK_KEY_RELEASE |
        XCB_EVENT_MASK_BUTTON_PRESS |
        XCB_EVENT_MASK_BUTTON_RELEASE |
        XCB_EVENT_MASK_POINTER_MOTION |
        XCB_EVENT_MASK_STRUCTURE_NOTIFY |
        XCB_EVENT_MASK_FOCUS_CHANGE;

    uint32_t value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
    uint32_t value_list[2] = {screen_->black_pixel, event_mask};

    xcb_create_window(
        connection_,
        XCB_COPY_FROM_PARENT,
        window_,
        screen_->root,
        0, 0,
        desc.width, desc.height,
        0,
        XCB_WINDOW_CLASS_INPUT_OUTPUT,
        screen_->root_visual,
        value_mask,
        value_list
    );

    width_ = desc.width;
    height_ = desc.height;

    // Set window title
    xcb_change_property(
        connection_,
        XCB_PROP_MODE_REPLACE,
        window_,
        XCB_ATOM_WM_NAME,
        XCB_ATOM_STRING,
        8,
        strlen(desc.title),
        desc.title
    );

    // Set up window close handling
    xcb_intern_atom_cookie_t protocols_cookie = xcb_intern_atom(
        connection_, 1, 12, "WM_PROTOCOLS");
    xcb_intern_atom_cookie_t delete_cookie = xcb_intern_atom(
        connection_, 0, 16, "WM_DELETE_WINDOW");

    xcb_intern_atom_reply_t* protocols_reply = xcb_intern_atom_reply(
        connection_, protocols_cookie, nullptr);
    xcb_intern_atom_reply_t* delete_reply = xcb_intern_atom_reply(
        connection_, delete_cookie, nullptr);

    if (protocols_reply && delete_reply) {
        wm_protocols_atom_ = protocols_reply->atom;
        wm_delete_window_atom_ = delete_reply->atom;

        xcb_change_property(
            connection_,
            XCB_PROP_MODE_REPLACE,
            window_,
            wm_protocols_atom_,
            XCB_ATOM_ATOM,
            32,
            1,
            &wm_delete_window_atom_
        );
    }

    free(protocols_reply);
    free(delete_reply);

    // Initialize key symbols for keycode translation
    key_symbols_ = xcb_key_symbols_alloc(connection_);

    // Map (show) the window
    xcb_map_window(connection_, window_);
    xcb_flush(connection_);

    open_ = true;
    return true;
}

void XcbWindow::destroy() {
    if (key_symbols_) {
        xcb_key_symbols_free(key_symbols_);
        key_symbols_ = nullptr;
    }

    if (window_) {
        xcb_destroy_window(connection_, window_);
        window_ = 0;
    }

    if (connection_) {
        xcb_disconnect(connection_);
        connection_ = nullptr;
    }

    open_ = false;
}

void XcbWindow::set_title(const char* title) {
    if (!connection_ || !window_) return;

    xcb_change_property(
        connection_,
        XCB_PROP_MODE_REPLACE,
        window_,
        XCB_ATOM_WM_NAME,
        XCB_ATOM_STRING,
        8,
        strlen(title),
        title
    );
    xcb_flush(connection_);
}

void XcbWindow::set_size(uint32_t width, uint32_t height) {
    if (!connection_ || !window_) return;

    uint32_t values[2] = {width, height};
    xcb_configure_window(
        connection_,
        window_,
        XCB_CONFIG_WINDOW_WIDTH | XCB_CONFIG_WINDOW_HEIGHT,
        values
    );
    xcb_flush(connection_);
}

void XcbWindow::poll_events() {
    xcb_generic_event_t* event;
    while ((event = xcb_poll_for_event(connection_))) {
        handle_event(event);
        free(event);
    }
}

void XcbWindow::wait_events() {
    xcb_generic_event_t* event = xcb_wait_for_event(connection_);
    if (event) {
        handle_event(event);
        free(event);
    }

    // Also process any other pending events
    poll_events();
}

void XcbWindow::handle_event(xcb_generic_event_t* event) {
    uint8_t event_type = event->response_type & ~0x80;

    switch (event_type) {
        case XCB_CONFIGURE_NOTIFY: {
            auto* configure = reinterpret_cast<xcb_configure_notify_event_t*>(event);
            if (configure->width != width_ || configure->height != height_) {
                width_ = configure->width;
                height_ = configure->height;
                if (callbacks_.on_resize) {
                    callbacks_.on_resize(width_, height_);
                }
            }
            break;
        }

        case XCB_KEY_PRESS:
        case XCB_KEY_RELEASE: {
            auto* key_event = reinterpret_cast<xcb_key_press_event_t*>(event);
            bool pressed = (event_type == XCB_KEY_PRESS);

            xcb_keysym_t keysym = xcb_key_symbols_get_keysym(
                key_symbols_, key_event->detail, 0);
            KeyCode key = xcb_keycode_to_keycode(keysym);
            Modifiers mods = xcb_state_to_modifiers(key_event->state);

            update_key_state(key, pressed);

            if (callbacks_.on_key) {
                callbacks_.on_key(key, pressed, mods);
            }

            // Generate character event for printable keys on press
            if (pressed && callbacks_.on_char) {
                if (keysym >= 0x20 && keysym <= 0x7E) {
                    callbacks_.on_char(keysym);
                }
            }
            break;
        }

        case XCB_BUTTON_PRESS:
        case XCB_BUTTON_RELEASE: {
            auto* button_event = reinterpret_cast<xcb_button_press_event_t*>(event);
            bool pressed = (event_type == XCB_BUTTON_PRESS);
            Modifiers mods = xcb_state_to_modifiers(button_event->state);

            // Buttons 1-3 are left, middle, right
            // Buttons 4-5 are scroll wheel
            if (button_event->detail >= 1 && button_event->detail <= 3) {
                MouseButton button;
                switch (button_event->detail) {
                    case 1: button = MouseButton::Left; break;
                    case 2: button = MouseButton::Middle; break;
                    case 3: button = MouseButton::Right; break;
                    default: button = MouseButton::Left; break;
                }

                mouse_button_states_[static_cast<uint8_t>(button)] = pressed;

                if (callbacks_.on_mouse_button) {
                    callbacks_.on_mouse_button(
                        button, pressed,
                        button_event->event_x, button_event->event_y,
                        mods
                    );
                }
            } else if (button_event->detail == 4 || button_event->detail == 5) {
                // Scroll wheel
                if (pressed && callbacks_.on_scroll) {
                    float dy = (button_event->detail == 4) ? 1.0f : -1.0f;
                    callbacks_.on_scroll(0, dy);
                }
            }
            break;
        }

        case XCB_MOTION_NOTIFY: {
            auto* motion = reinterpret_cast<xcb_motion_notify_event_t*>(event);
            mouse_x_ = motion->event_x;
            mouse_y_ = motion->event_y;

            if (callbacks_.on_mouse_move) {
                callbacks_.on_mouse_move(mouse_x_, mouse_y_);
            }
            break;
        }

        case XCB_FOCUS_IN:
            focused_ = true;
            if (callbacks_.on_focus) {
                callbacks_.on_focus(true);
            }
            break;

        case XCB_FOCUS_OUT:
            focused_ = false;
            if (callbacks_.on_focus) {
                callbacks_.on_focus(false);
            }
            break;

        case XCB_CLIENT_MESSAGE: {
            auto* client = reinterpret_cast<xcb_client_message_event_t*>(event);
            if (client->data.data32[0] == wm_delete_window_atom_) {
                open_ = false;
                if (callbacks_.on_close) {
                    callbacks_.on_close();
                }
            }
            break;
        }

        case XCB_UNMAP_NOTIFY:
            minimized_ = true;
            break;

        case XCB_MAP_NOTIFY:
            minimized_ = false;
            break;

        default:
            break;
    }
}

void XcbWindow::update_key_state(KeyCode key, bool pressed) {
    uint16_t index = static_cast<uint16_t>(key);
    if (index < 512) {
        key_states_[index] = pressed;
    }
}

bool XcbWindow::is_key_pressed(KeyCode key) const {
    uint16_t index = static_cast<uint16_t>(key);
    if (index < 512) {
        return key_states_[index];
    }
    return false;
}

bool XcbWindow::is_mouse_button_pressed(MouseButton button) const {
    uint8_t index = static_cast<uint8_t>(button);
    if (index < 8) {
        return mouse_button_states_[index];
    }
    return false;
}

void XcbWindow::get_mouse_position(int& x, int& y) const {
    x = mouse_x_;
    y = mouse_y_;
}

// Factory implementation
std::unique_ptr<Window> Window::create(const WindowDesc& desc) {
    auto window = std::make_unique<XcbWindow>();
    if (window->initialize(desc)) {
        return window;
    }
    return nullptr;
}

} // namespace viz::platform

#endif // VIZ_PLATFORM_LINUX
