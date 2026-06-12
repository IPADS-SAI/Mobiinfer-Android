#pragma once

#include <string>
#include <string_view>
#include <memory>
#include <mutex>
#include <functional>

struct omni_context;

enum class SessionMode {
    FULL_DUPLEX,
    TURN_BASED,
};

enum class SessionState {
    UNINITIALIZED,
    ACTIVE,
    CLOSED,
};

struct OmniSession {
    std::string session_id;
    SessionState state = SessionState::UNINITIALIZED;
    SessionMode mode = SessionMode::FULL_DUPLEX;
    omni_context * octx = nullptr;
    bool owns_octx = false;
    double created_at = 0.0;
    std::function<void()> close_ws;
};

// SessionManager — manages the single active backend session.
// Protocol constraint: at most one active session per backend.
class SessionManager {
public:
    ~SessionManager();

    // Allocate a new session (UNINITIALIZED state). Returns session_id.
    // Fails if an active session already exists.
    std::string allocate();

    // Activate a session after successful init. Sets octx pointer.
    bool activate(const std::string & session_id, omni_context * octx, bool owns_octx);

    // Get a session by id. Returns nullptr if not found.
    OmniSession * get(const std::string & session_id);

    // Register/trigger transport close without holding the manager lock while
    // invoking the callback.
    void set_close_callback(const std::string & session_id, std::function<void()> cb);
    void request_transport_close(const std::string & session_id);

    // Close and forget a session. Releases omni_context if owned.
    void close(const std::string & session_id);

    // Handle WS disconnect — auto-close active session.
    void on_disconnect();

    // Check if an active session exists.
    bool has_active() const;

    // Close all sessions (shutdown cleanup).
    void shutdown();

private:
    mutable std::mutex mtx_;
    std::unique_ptr<OmniSession> active_;
    std::string generate_uuid();
};
