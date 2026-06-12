#include "session.h"
#include "omni.h"

#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

SessionManager::~SessionManager() {
    shutdown();
}

static double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

std::string SessionManager::generate_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist;

    uint64_t a = dist(gen);
    uint64_t b = dist(gen);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << a
        << std::setw(16) << b;
    return oss.str();
}

std::string SessionManager::allocate() {
    std::lock_guard<std::mutex> lock(mtx_);

    if (active_ && active_->state != SessionState::CLOSED) {
        return ""; // fail-fast: only one active session
    }

    auto session = std::make_unique<OmniSession>();
    session->session_id = generate_uuid();
    session->state = SessionState::UNINITIALIZED;
    session->created_at = now_seconds();

    std::string id = session->session_id;
    active_ = std::move(session);
    return id;
}

bool SessionManager::activate(const std::string & session_id, omni_context * octx, bool owns_octx) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!active_ || active_->session_id != session_id) {
        return false;
    }
    if (active_->state != SessionState::UNINITIALIZED) {
        return false; // already active or closed
    }

    active_->octx = octx;
    active_->owns_octx = owns_octx;
    active_->state = SessionState::ACTIVE;
    return true;
}

OmniSession * SessionManager::get(const std::string & session_id) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!active_ || active_->session_id != session_id) {
        return nullptr;
    }
    return active_.get();
}

void SessionManager::set_close_callback(const std::string & session_id, std::function<void()> cb) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!active_ || active_->session_id != session_id) {
        return;
    }
    active_->close_ws = std::move(cb);
}

void SessionManager::request_transport_close(const std::string & session_id) {
    std::function<void()> close_ws;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!active_ || active_->session_id != session_id) {
            return;
        }
        close_ws = active_->close_ws;
    }

    if (close_ws) {
        close_ws();
    }
}

void SessionManager::close(const std::string & session_id) {
    std::unique_ptr<OmniSession> to_free;
    {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!active_ || active_->session_id != session_id) {
            return; // not found — caller gets 404
        }

        if (active_->state == SessionState::ACTIVE) {
            active_->state = SessionState::CLOSED;
        }

        to_free = std::move(active_);
    }

    // Release omni_context outside the manager lock. omni_free stops and joins
    // inference threads and can touch shared backend state, so holding mtx_ here
    // can race with WS cleanup or block unrelated lifecycle checks.
    if (to_free && to_free->owns_octx && to_free->octx) {
        omni_free(to_free->octx);
        to_free->octx = nullptr;
    }
}

void SessionManager::on_disconnect() {
    std::unique_ptr<OmniSession> to_free;
    {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!active_ || active_->state == SessionState::CLOSED) {
            return;
        }

        active_->state = SessionState::CLOSED;
        to_free = std::move(active_);
    }

    if (to_free && to_free->owns_octx && to_free->octx) {
        omni_free(to_free->octx);
        to_free->octx = nullptr;
    }
}

bool SessionManager::has_active() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_ && active_->state == SessionState::ACTIVE;
}

void SessionManager::shutdown() {
    std::unique_ptr<OmniSession> to_free;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        to_free = std::move(active_);
    }

    if (to_free && to_free->owns_octx && to_free->octx) {
        omni_free(to_free->octx);
    }
}
