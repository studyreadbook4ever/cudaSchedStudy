// SPDX-License-Identifier: Unlicense
#include "gpu_backend.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <poll.h>
#include <set>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using J = std::string;
static volatile sig_atomic_t stop_requested = 0;
static void on_signal(int) { stop_requested = 1; }
static J str(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 32)
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
      else
        out << c;
    }
  }
  out << '"';
  return out.str();
}
template <class T> static J num(T value) {
  char buffer[128];
  auto converted = [&] {
    if constexpr (std::is_floating_point_v<T>)
      return std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general, 12);
    else
      return std::to_chars(buffer, buffer + sizeof(buffer), value);
  }();
  if (converted.ec != std::errc{})
    throw std::runtime_error("JSON number formatting");
  return J(buffer, converted.ptr);
}
static J boolean(bool value) { return value ? "true" : "false"; }
static J object(std::initializer_list<std::pair<std::string, J>> fields) {
  J out = "{";
  bool first = true;
  for (const auto &[key, value] : fields) {
    if (!first)
      out += ',';
    first = false;
    out += str(key) + ":" + value;
  }
  return out + '}';
}
static J array(const std::vector<J> &values) {
  J out = "[";
  bool first = true;
  for (const auto &value : values) {
    if (!first)
      out += ',';
    first = false;
    out += value;
  }
  return out + ']';
}
static J error(const std::string &code) { return object({{"ok", "false"}, {"error", str(code)}}); }
static bool constant_equal(const std::string &a, const std::string &b) {
  unsigned diff = unsigned(a.size() ^ b.size());
  for (std::size_t i = 0; i < std::max(a.size(), b.size()); ++i)
    diff |= unsigned(i < a.size() ? a[i] : 0) ^ unsigned(i < b.size() ? b[i] : 0);
  return diff == 0;
}
static std::uint64_t integer(const std::string &value, std::uint64_t low, std::uint64_t high) {
  if (value.empty() || value.size() > 20 ||
      !std::all_of(value.begin(), value.end(),
                   [](unsigned char c) { return c >= '0' && c <= '9'; }))
    throw std::invalid_argument("integer");
  std::size_t used = 0;
  auto result = std::stoull(value, &used);
  if (used != value.size() || result < low || result > high)
    throw std::invalid_argument("range");
  return result;
}
static std::string random_token() {
  unsigned char bytes[32];
  std::size_t offset = 0;
  while (offset < sizeof(bytes)) {
    auto n = getrandom(bytes + offset, sizeof(bytes) - offset, 0);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("getrandom");
    offset += std::size_t(n);
  }
  std::ostringstream out;
  for (auto b : bytes)
    out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
  return out.str();
}
static std::string admin_token(const fs::path &path) {
  if (fs::exists(path)) {
    struct stat st{};
    if (lstat(path.c_str(), &st) < 0 || !S_ISREG(st.st_mode) || st.st_uid != getuid() ||
        (st.st_mode & 077) != 0)
      throw std::runtime_error("admin token must be an owned regular file mode 0600");
    std::ifstream input(path);
    std::string token;
    input >> token;
    if (token.size() != 64)
      throw std::runtime_error("invalid admin token file");
    return token;
  }
  auto token = random_token();
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0)
    throw std::runtime_error("create admin token: " + std::string(strerror(errno)));
  auto body = token + '\n';
  std::size_t offset = 0;
  while (offset < body.size()) {
    auto n = write(fd, body.data() + offset, body.size() - offset);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      close(fd);
      throw std::runtime_error("write admin token");
    }
    offset += std::size_t(n);
  }
  close(fd);
  return token;
}
struct Config {
  std::string socket = "runtime/socket/gpu.sock", admin = "runtime/private/admin.token", trace;
  gpu::BackendConfig backend;
  std::uint32_t quantum = 1;
  std::uint64_t batch_elements = 524288;
  unsigned tick_ms = 2;
  unsigned max_clients = 128;
};
struct Tenant {
  std::string name, token;
  unsigned quota_slots = 4, weight = 1;
  bool enabled = true;
  bool history_dirty = false;
  bool waiting_for_lane = false;
  int lane = -1;
  std::set<std::uint64_t> sessions;
  std::set<std::uint64_t> job_ids, live_jobs;
  std::deque<std::uint64_t> ready;
  std::deque<std::uint64_t> terminal_history;
  std::uint64_t submitted = 0, completed = 0, cancelled = 0, scheduled = 0, completed_steps = 0;
  std::uint64_t credit = 0;
  unsigned live_slots = 0;
};
struct Job {
  std::uint64_t id = 0, owner = 0;
  Tenant *tenant = nullptr;
  int slot = -1, lane = -1;
  std::size_t elements = 0;
  std::uint32_t total = 0, done = 0, seed = 0, rounds = 1;
  bool started = false, inflight = false, cancel = false;
  std::string state = "queued";
  gpu::BatchResult result;
  double gpu_ms = 0;
  std::uint64_t batch_id = 0;
  std::uint64_t terminal_order = 0;
  double created_ms = 0, finished_ms = 0;
  bool terminal() const { return terminal_order != 0; }
};
struct Client {
  int fd = -1;
  std::uint64_t id = 0;
  Tenant *tenant = nullptr;
  std::string input, output;
  bool closing = false;
  bool admin_authenticated = false;
  bool read_eof = false, close_timer_started = false;
  Clock::time_point created = Clock::now(), last_activity = Clock::now();
  Clock::time_point closing_since;
};
struct Flight {
  std::uint64_t batch_id = 0;
  std::vector<std::pair<std::uint64_t, std::uint32_t>> jobs;
  std::uint64_t operations = 0;
};
class Server {
  Config cfg;
  gpu::Backend backend;
  int listener = -1;
  std::string admin;
  std::map<int, Client> clients;
  std::map<std::string, Tenant> tenants;
  std::map<std::uint64_t, Job> jobs;
  std::vector<Tenant *> lanes;
  std::vector<Flight> flights;
  std::deque<std::string> waiting_tenants;
  std::uint64_t next_client = 1, next_job = 1, batches = 0, multi_job_batches = 0, scheduled = 0;
  std::uint64_t next_terminal = 1;
  bool draining = false;
  Clock::time_point origin = Clock::now(), last_tick = Clock::now();
  std::ofstream trace;
  double now_ms() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
  }
  std::uint64_t slot_bytes() const { return cfg.backend.max_elements_per_slot * sizeof(float); }
  J device_json() const {
    const auto &d = backend.info();
    std::vector<J> counts;
    for (auto count : d.sm_counts)
      counts.push_back(num(count));
    return object({{"mode", str(d.mode)},
                   {"name", str(d.name)},
                   {"sm_count", num(d.sm_count)},
                   {"sm_counts", array(counts)},
                   {"lanes", num(d.lanes)},
                   {"cc_major", num(d.cc_major)},
                   {"cc_minor", num(d.cc_minor)},
                   {"total_memory", num(d.total_memory)},
                   {"pool_bytes", num(d.pool_bytes)},
                   {"pinned_bytes", num(d.pinned_bytes)},
                   {"max_elements", num(cfg.backend.max_elements_per_slot)},
                   {"slot_bytes", num(slot_bytes())}});
  }
  J job_json(const Job &j) const {
    std::vector<J> samples;
    if (j.state == "done")
      for (unsigned i = 0; i < j.result.sample_count; ++i)
        samples.push_back(num(j.result.sample_values[i]));
    return object({{"job_id", num(j.id)},
                   {"tenant", str(j.tenant->name)},
                   {"state", str(j.state)},
                   {"completed_steps", num(j.done)},
                   {"total_steps", num(j.total)},
                   {"elements", num(j.elements)},
                   {"seed", num(j.seed)},
                   {"rounds", num(j.rounds)},
                   {"cancel_requested", boolean(j.cancel)},
                   {"inflight", boolean(j.inflight)},
                   {"slot", num(j.slot)},
                   {"lane", num(j.lane)},
                   {"waiting_for_lane", boolean(!j.terminal() && j.lane < 0)},
                   {"reserved_bytes", num(j.terminal() ? 0 : slot_bytes())},
                   {"allocated_bytes", num(j.slot < 0 ? 0 : slot_bytes())},
                   {"samples", array(samples)},
                   {"first", j.state == "done" ? num(j.result.first) : "null"},
                   {"last", j.state == "done" ? num(j.result.last) : "null"},
                   {"sample_checksum", j.state == "done" ? num(j.result.sample_checksum) : "null"},
                   {"gpu_ms", num(j.gpu_ms)},
                   {"sm_mask", num(j.result.sm_mask)},
                   {"created_ms", num(j.created_ms)},
                   {"finished_ms", num(j.finished_ms)},
                   {"batch_id", num(j.batch_id)}});
  }
  J tenant_json(const Tenant &t, bool with_jobs = true) const {
    std::vector<J> own;
    if (with_jobs)
      for (auto id : t.job_ids)
        own.push_back(job_json(jobs.at(id)));
    return object({{"tenant", str(t.name)},
                   {"name", str(t.name)},
                   {"enabled", boolean(t.enabled)},
                   {"lane", num(t.lane)},
                   {"waiting_for_lane", boolean(t.waiting_for_lane)},
                   {"quota_slots", num(t.quota_slots)},
                   {"quota_bytes", num(t.quota_slots * slot_bytes())},
                   {"live_slots", num(t.live_slots)},
                   {"reserved_bytes", num(t.live_slots * slot_bytes())},
                   {"allocated_bytes", num(t.lane < 0 ? 0 : t.live_slots * slot_bytes())},
                   {"weight", num(t.weight)},
                   {"submitted", num(t.submitted)},
                   {"completed", num(t.completed)},
                   {"cancelled", num(t.cancelled)},
                   {"scheduled_elements", num(t.scheduled)},
                   {"completed_steps", num(t.completed_steps)},
                   {"active_sessions", num(t.sessions.size())},
                   {"jobs", array(own)}});
  }
  J global_json() const {
    unsigned live = 0, resident = 0, inflight = 0;
    for (const auto &[_, t] : tenants) {
      live += t.live_slots;
      if (t.lane >= 0)
        resident += t.live_slots;
    }
    for (unsigned lane = 0; lane < lanes.size(); ++lane)
      if (backend.busy(lane))
        ++inflight;
    return object({{"batches", num(batches)},
                   {"multi_job_batches", num(multi_job_batches)},
                   {"inflight_batches", num(inflight)},
                   {"live_slots", num(live)},
                   {"resident_slots", num(resident)},
                   {"waiting_tenants", num(waiting_tenants.size())},
                   {"reserved_bytes", num(live * slot_bytes())},
                   {"allocated_bytes", num(resident * slot_bytes())},
                   {"scheduled_elements", num(scheduled)},
                   {"backend", device_json()}});
  }
  void event(const J &record) {
    if (trace) {
      trace << record << '\n';
      trace.flush();
    }
  }
  void trim_history(Tenant &tenant) {
    while (tenant.terminal_history.size() > 128) {
      const auto id = tenant.terminal_history.front();
      tenant.terminal_history.pop_front();
      tenant.job_ids.erase(id);
      jobs.erase(id);
    }
    tenant.history_dirty = false;
  }
  void trim_pending_history() {
    for (auto &[_, tenant] : tenants)
      if (tenant.history_dirty)
        trim_history(tenant);
  }
  void release_lane(Tenant &t) {
    if (t.live_slots == 0) {
      if (t.waiting_for_lane) {
        std::erase(waiting_tenants, t.name);
        t.waiting_for_lane = false;
      }
      t.ready.clear();
    }
    if (t.lane >= 0 && t.live_slots == 0 && !backend.busy(unsigned(t.lane))) {
      event(object(
          {{"event", str("lane_released")}, {"tenant", str(t.name)}, {"lane", num(t.lane)}}));
      lanes[std::size_t(t.lane)] = nullptr;
      t.lane = -1;
      t.credit = 0;
      t.ready.clear();
    }
  }
  void wait_for_lane(Tenant &t) {
    if (t.lane < 0 && !t.waiting_for_lane && t.live_slots > 0) {
      t.waiting_for_lane = true;
      waiting_tenants.push_back(t.name);
    }
  }
  void admit_waiters() {
    if (draining)
      return;
    while (!waiting_tenants.empty()) {
      auto free_lane = std::find(lanes.begin(), lanes.end(), nullptr);
      if (free_lane == lanes.end())
        break;
      const auto name = waiting_tenants.front();
      waiting_tenants.pop_front();
      auto found = tenants.find(name);
      if (found == tenants.end())
        continue;
      auto &t = found->second;
      t.waiting_for_lane = false;
      if (!t.enabled || t.live_slots == 0 || t.sessions.empty())
        continue;
      t.lane = int(free_lane - lanes.begin());
      *free_lane = &t;
      for (auto id : t.live_jobs) {
        auto &j = jobs.at(id);
        j.slot = backend.allocate(unsigned(t.lane), j.elements, j.seed);
        j.lane = t.lane;
      }
      event(object(
          {{"event", str("lane_assigned")}, {"tenant", str(t.name)}, {"lane", num(t.lane)}}));
    }
  }
  void finish(Job &j, const std::string &state) {
    if (j.terminal())
      return;
    if (j.slot >= 0)
      backend.release(unsigned(j.tenant->lane), j.slot);
    j.slot = -1;
    --j.tenant->live_slots;
    j.tenant->live_jobs.erase(j.id);
    std::erase(j.tenant->ready, j.id);
    j.inflight = false;
    j.state = state;
    j.finished_ms = now_ms();
    j.terminal_order = next_terminal++;
    j.tenant->terminal_history.push_back(j.id);
    j.tenant->history_dirty = true;
    if (state == "done")
      ++j.tenant->completed;
    else
      ++j.tenant->cancelled;
    event(object({{"event", str("job_terminal")},
                  {"job_id", num(j.id)},
                  {"tenant", str(j.tenant->name)},
                  {"state", str(state)},
                  {"completed_steps", num(j.done)}}));
  }
  void cancel_session(Client &c) {
    if (!c.tenant)
      return;
    auto &t = *c.tenant;
    t.sessions.erase(c.id);
    const std::vector<std::uint64_t> live(t.live_jobs.begin(), t.live_jobs.end());
    for (auto id : live) {
      auto &j = jobs.at(id);
      if (j.owner == c.id) {
        j.cancel = true;
        if (!j.inflight && (t.lane < 0 || !backend.busy(unsigned(t.lane))))
          finish(j, "cancelled");
      }
    }
    event(object({{"event", str("leave")}, {"tenant", str(t.name)}, {"session_id", num(c.id)}}));
    c.tenant = nullptr;
    release_lane(t);
  }
  void disconnect(int fd) {
    auto it = clients.find(fd);
    if (it == clients.end())
      return;
    cancel_session(it->second);
    close(fd);
    clients.erase(it);
  }
  std::vector<std::string> words(const std::string &line) const {
    if (line.size() > 4096 ||
        std::any_of(line.begin(), line.end(), [](unsigned char c) { return c < 32 || c > 126; }))
      throw std::invalid_argument("line");
    std::istringstream input(line);
    std::vector<std::string> out;
    std::string token;
    while (input >> token) {
      if (out.size() >= 8)
        throw std::invalid_argument("fields");
      out.push_back(token);
    }
    if (out.empty())
      throw std::invalid_argument("empty");
    return out;
  }
  J command(Client &c, const std::string &line) {
    // Trim at a boundary with no live Job references: cancellation must still
    // be able to return the job it just made terminal, including an old ID.
    trim_pending_history();
    auto w = words(line);
    const auto &op = w[0];
    if (op == "PING" && w.size() == 1)
      return object({{"ok", "true"}, {"ready", boolean(!draining)}, {"backend", device_json()}});
    if (op == "ADMIN") {
      if (w.size() < 3)
        return error("invalid_request");
      if (!constant_equal(w[1], admin))
        return error("unauthorized");
      c.admin_authenticated = true;
      if (w[2] == "PROVISION" && w.size() == 6) {
        if (draining)
          return error("server_draining");
        const auto &name = w[3];
        if (name.size() > 48 || !std::all_of(name.begin(), name.end(), [](unsigned char x) {
              return std::isalnum(x) || x == '_' || x == '-';
            }))
          return error("invalid_request");
        if (tenants.contains(name))
          return error("already_exists");
        if (tenants.size() >= 128)
          return error("capacity");
        Tenant t;
        t.name = name;
        t.token = random_token();
        t.quota_slots = unsigned(integer(w[4], 1, cfg.backend.slots_per_lane));
        t.weight = unsigned(integer(w[5], 1, 16));
        auto [it, _] = tenants.emplace(name, std::move(t));
        return object({{"ok", "true"},
                       {"name", str(name)},
                       {"token", str(it->second.token)},
                       {"quota_slots", num(it->second.quota_slots)},
                       {"quota_bytes", num(it->second.quota_slots * slot_bytes())},
                       {"weight", num(it->second.weight)}});
      }
      if (w[2] == "STATS" && w.size() == 3) {
        std::vector<J> ts;
        for (const auto &[_, t] : tenants)
          ts.push_back(tenant_json(t, false));
        return object({{"ok", "true"},
                       {"global", global_json()},
                       {"tenants", array(ts)},
                       {"jobs_omitted", "true"}});
      }
      if (w[2] == "INSPECT" && w.size() == 4) {
        auto found = tenants.find(w[3]);
        if (found == tenants.end())
          return error("not_found");
        return object({{"ok", "true"}, {"tenant", tenant_json(found->second)}});
      }
      if (w[2] == "REVOKE" && w.size() == 4) {
        auto it = tenants.find(w[3]);
        if (it == tenants.end())
          return error("not_found");
        auto &t = it->second;
        t.enabled = false;
        for (auto &[_, cl] : clients)
          if (cl.tenant == &t) {
            cancel_session(cl);
            cl.output += error("revoked") + '\n';
            cl.closing = true;
          }
        release_lane(t);
        return object({{"ok", "true"}, {"revoked", str(t.name)}});
      }
      if (w[2] == "FORGET" && w.size() == 4) {
        auto found = tenants.find(w[3]);
        if (found == tenants.end())
          return error("not_found");
        auto &t = found->second;
        // Keep pointers held by sessions and in-flight jobs valid until drain.
        if (t.enabled || !t.sessions.empty() || t.live_slots != 0 || t.lane >= 0)
          return error("tenant_busy");
        const auto name = t.name;
        std::erase(waiting_tenants, name);
        for (auto id : t.job_ids)
          jobs.erase(id);
        tenants.erase(found);
        return object({{"ok", "true"}, {"forgotten", str(name)}});
      }
      if (w[2] == "SHUTDOWN" && w.size() == 3) {
        draining = true;
        return object({{"ok", "true"}, {"draining", "true"}});
      }
      return error("invalid_request");
    }
    if (op == "HELLO" && w.size() == 2) {
      if (c.tenant)
        return error("already_authenticated");
      if (draining)
        return error("server_draining");
      Tenant *found = nullptr;
      for (auto &[_, t] : tenants)
        if (constant_equal(w[1], t.token) && t.enabled)
          found = &t;
      if (!found)
        return error("unauthorized");
      c.tenant = found;
      found->sessions.insert(c.id);
      event(object({{"event", str("join")},
                    {"tenant", str(found->name)},
                    {"session_id", num(c.id)},
                    {"lane", num(found->lane)}}));
      return object({{"ok", "true"},
                     {"tenant", str(found->name)},
                     {"session_id", num(c.id)},
                     {"lane", num(found->lane)},
                     {"quota_slots", num(found->quota_slots)},
                     {"quota_bytes", num(found->quota_slots * slot_bytes())},
                     {"weight", num(found->weight)}});
    }
    if (!c.tenant)
      return error("not_authenticated");
    auto &t = *c.tenant;
    if (op == "HEARTBEAT" && w.size() == 1)
      return object({{"ok", "true"}, {"tenant", str(t.name)}, {"lane", num(t.lane)}});
    if (op == "SUBMIT" && w.size() == 5) {
      if (draining)
        return error("server_draining");
      auto elements = integer(w[1], 1, cfg.backend.max_elements_per_slot);
      auto steps = integer(w[2], 1, 100000);
      auto seed = integer(w[3], 0, std::numeric_limits<std::uint32_t>::max());
      auto rounds = integer(w[4], 1, 256);
      if (t.live_slots >= t.quota_slots)
        return error("quota_exceeded");
      Job j;
      j.id = next_job++;
      j.owner = c.id;
      j.tenant = &t;
      j.elements = std::size_t(elements);
      j.total = std::uint32_t(steps);
      j.seed = std::uint32_t(seed);
      j.rounds = std::uint32_t(rounds);
      j.created_ms = now_ms();
      if (t.lane >= 0)
        j.slot = backend.allocate(unsigned(t.lane), j.elements, j.seed);
      j.lane = t.lane;
      const auto id = j.id;
      jobs.emplace(id, std::move(j));
      t.job_ids.insert(id);
      t.live_jobs.insert(id);
      t.ready.push_back(id);
      ++t.live_slots;
      ++t.submitted;
      wait_for_lane(t);
      admit_waiters();
      trim_history(t);
      event(object({{"event", str("submit")},
                    {"job_id", num(id)},
                    {"tenant", str(t.name)},
                    {"elements", num(elements)},
                    {"steps", num(steps)}}));
      return object({{"ok", "true"},
                     {"job_id", num(id)},
                     {"state", str("queued")},
                     {"waiting_for_lane", boolean(t.lane < 0)},
                     {"reserved_bytes", num(slot_bytes())}});
    }
    if ((op == "GET" || op == "CANCEL") && w.size() == 2) {
      auto id = integer(w[1], 1, std::numeric_limits<std::uint64_t>::max());
      auto it = jobs.find(id);
      if (it == jobs.end() || it->second.tenant != &t)
        return error("not_found");
      auto &j = it->second;
      if (op == "CANCEL" && !j.terminal()) {
        j.cancel = true;
        if (!j.inflight && (t.lane < 0 || !backend.busy(unsigned(t.lane))))
          finish(j, "cancelled");
        release_lane(t);
      }
      return object(
          {{"ok", "true"}, {"job", job_json(j)}, {"cancellation_requested", boolean(j.cancel)}});
    }
    if (op == "LIST" && w.size() == 1) {
      std::vector<J> own;
      for (auto id : t.job_ids)
        own.push_back(job_json(jobs.at(id)));
      return object({{"ok", "true"}, {"jobs", array(own)}});
    }
    if (op == "STATS" && w.size() == 1)
      return object({{"ok", "true"}, {"tenant", tenant_json(t)}, {"global", global_json()}});
    if (op == "LEAVE" && w.size() == 1) {
      cancel_session(c);
      c.closing = true;
      return object({{"ok", "true"}, {"bye", "true"}});
    }
    return error("invalid_request");
  }
  void accept_clients() {
    for (unsigned attempts = 0; attempts < 32; ++attempts) {
      int fd = accept4(listener, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
          return;
        if (errno == EINTR)
          continue;
        throw std::runtime_error("accept");
      }
      if (clients.size() >= cfg.max_clients || draining) {
        close(fd);
        continue;
      }
      Client c;
      c.fd = fd;
      c.id = next_client++;
      clients.emplace(fd, std::move(c));
    }
  }
  void read_client(Client &c) {
    char buffer[4096];
    std::size_t received = 0;
    while (!c.read_eof && received < 16384) {
      auto n = recv(c.fd, buffer, sizeof(buffer), 0);
      if (n == 0) {
        c.read_eof = true;
        break;
      }
      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
          break;
        if (errno == EINTR)
          continue;
        c.closing = true;
        cancel_session(c);
        return;
      }
      received += std::size_t(n);
      c.input.append(buffer, std::size_t(n));
      c.last_activity = Clock::now();
      if (c.input.size() > 65536) {
        c.output = error("invalid_request") + '\n';
        c.closing = true;
        cancel_session(c);
        return;
      }
    }
    for (unsigned handled = 0; handled < 32 && !c.closing; ++handled) {
      auto pos = c.input.find('\n');
      if (pos == std::string::npos) {
        if (c.input.size() > 4096) {
          c.output += error("invalid_request") + '\n';
          c.closing = true;
          cancel_session(c);
        }
        break;
      }
      auto line = c.input.substr(0, pos);
      c.input.erase(0, pos + 1);
      try {
        c.output += command(c, line) + '\n';
      } catch (const std::invalid_argument &) {
        c.output += error("invalid_request") + '\n';
      } catch (const std::out_of_range &) {
        c.output += error("invalid_request") + '\n';
      }
      if (c.output.size() > 1048576) {
        c.closing = true;
        cancel_session(c);
        c.output.clear();
      }
    }
    if (c.read_eof && c.input.find('\n') == std::string::npos) {
      if (!c.input.empty())
        c.output += error("invalid_request") + '\n';
      c.input.clear();
      c.closing = true;
      cancel_session(c);
    }
  }
  void write_client(Client &c) {
    if (c.output.empty())
      return;
    auto n = send(c.fd, c.output.data(), c.output.size(), MSG_NOSIGNAL);
    if (n > 0)
      c.output.erase(0, std::size_t(n));
    else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      c.closing = true;
      cancel_session(c);
      c.output.clear();
    }
  }
  void retire() {
    for (unsigned lane = 0; lane < lanes.size(); ++lane) {
      auto completed = backend.poll(lane);
      if (!completed)
        continue;
      auto flight = std::move(flights[lane]);
      flights[lane] = {};
      for (const auto &[id, steps] : flight.jobs) {
        auto &j = jobs.at(id);
        auto found = std::find_if(completed->results.begin(), completed->results.end(),
                                  [&](const auto &r) { return r.slot == j.slot; });
        if (found == completed->results.end())
          throw std::runtime_error("missing batch result");
        if (found->summary_available)
          j.result = *found;
        j.started = true;
        j.inflight = false;
        j.done += steps;
        j.tenant->completed_steps += steps;
        const auto work = std::uint64_t(j.elements) * steps * j.rounds;
        j.gpu_ms += double(completed->gpu_ms) * double(work) / double(flight.operations);
        if (j.cancel || draining || !j.tenant->enabled || !j.tenant->sessions.contains(j.owner))
          finish(j, "cancelled");
        else if (j.done == j.total)
          finish(j, "done");
        else {
          j.state = "running";
          j.tenant->ready.push_back(id);
        }
      }
      event(object({{"event", str("batch_complete")},
                    {"batch_id", num(flight.batch_id)},
                    {"lane", num(lane)},
                    {"gpu_ms", num(completed->gpu_ms)},
                    {"operations", num(flight.operations)}}));
      if (lanes[lane]) {
        auto &t = *lanes[lane];
        const std::vector<std::uint64_t> live(t.live_jobs.begin(), t.live_jobs.end());
        for (auto id : live) {
          auto &j = jobs.at(id);
          if (j.cancel && !j.inflight)
            finish(j, "cancelled");
        }
        release_lane(t);
        trim_history(t);
      }
    }
    for (auto &[_, t] : tenants) {
      if (t.lane < 0 || !backend.busy(unsigned(t.lane))) {
        const std::vector<std::uint64_t> live(t.live_jobs.begin(), t.live_jobs.end());
        for (auto id : live) {
          auto &j = jobs.at(id);
          if (j.cancel || draining)
            finish(j, "cancelled");
        }
        release_lane(t);
      }
    }
    admit_waiters();
  }
  void schedule() {
    if (draining)
      return;
    auto now = Clock::now();
    const bool refill = now - last_tick >= std::chrono::milliseconds(cfg.tick_ms);
    if (refill)
      last_tick = now;
    unsigned total_weight = 0;
    for (auto &[_, t] : tenants)
      if (t.enabled && t.live_slots > 0 && !t.sessions.empty() && t.lane >= 0)
        total_weight += t.weight;
    if (total_weight == 0)
      return;
    for (auto &[_, t] : tenants) {
      if (!t.enabled || t.live_slots == 0 || t.sessions.empty() || t.lane < 0)
        continue;
      auto credit_limit =
          std::max(cfg.batch_elements * 8,
                   std::uint64_t(cfg.backend.max_elements_per_slot) * 256 * cfg.quantum);
      if (refill)
        t.credit = std::min(
            credit_limit,
            t.credit + std::max<std::uint64_t>(1, cfg.batch_elements * t.weight / total_weight));
      const auto lane = unsigned(t.lane);
      if (backend.busy(lane))
        continue;
      std::vector<gpu::BatchItem> batch;
      Flight flight;
      std::vector<J> ids;
      auto count = t.ready.size();
      std::uint64_t elements = 0;
      for (std::size_t k = 0; k < count; ++k) {
        auto id = t.ready.front();
        t.ready.pop_front();
        auto it = jobs.find(id);
        if (it == jobs.end() || it->second.slot < 0)
          continue;
        auto &j = it->second;
        if (j.cancel) {
          finish(j, "cancelled");
          continue;
        }
        auto steps = std::min(cfg.quantum, j.total - j.done);
        auto work = std::uint64_t(j.elements) * steps * j.rounds;
        if (t.credit < work) {
          // Reserve the next refill for this job. Skipping it allows a steady
          // stream of cheaper jobs to spend every refill and starve this one.
          t.ready.push_front(id);
          break;
        }
        if (!batch.empty() && elements + j.elements > cfg.batch_elements) {
          t.ready.push_back(id);
          continue;
        }
        t.credit -= work;
        t.scheduled += work;
        scheduled += work;
        elements += j.elements;
        const bool final_slice = j.done + steps == j.total;
        batch.push_back({j.slot, steps, !j.started, final_slice, j.rounds, final_slice});
        j.inflight = true;
        j.state = "running";
        flight.jobs.emplace_back(id, steps);
        flight.operations += work;
        ids.push_back(num(id));
      }
      if (!batch.empty()) {
        flight.batch_id = ++batches;
        if (batch.size() > 1)
          ++multi_job_batches;
        for (auto [id, _] : flight.jobs)
          jobs.at(id).batch_id = flight.batch_id;
        backend.submit(lane, batch);
        flights[lane] = flight;
        event(object({{"event", str("batch_submit")},
                      {"batch_id", num(flight.batch_id)},
                      {"lane", num(lane)},
                      {"tenant", str(t.name)},
                      {"job_ids", array(ids)},
                      {"total_elements", num(elements)},
                      {"operations", num(flight.operations)}}));
      }
    }
  }
  int poll_timeout() const {
    int timeout = 1000;
    for (unsigned lane = 0; lane < lanes.size(); ++lane) {
      if (backend.busy(lane)) {
        timeout = std::min(timeout, 1);
        continue;
      }
      if (lanes[lane] && lanes[lane]->live_slots > 0) {
        const auto elapsed = Clock::now() - last_tick;
        const auto remaining = std::chrono::milliseconds(cfg.tick_ms) - elapsed;
        // Round up so an idle lane waiting for credit does not busy-spin.
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(remaining).count();
        timeout = std::min(timeout, int(std::max<std::int64_t>(0, (us + 999) / 1000)));
      }
    }
    for (const auto &[_, c] : clients) {
      if (!c.closing && c.input.find('\n') != std::string::npos)
        return 0;
      if (c.closing)
        timeout = std::min(timeout, 100);
      if (!c.closing && !c.tenant && !c.admin_authenticated) {
        const auto age = Clock::now() - c.created;
        const auto remaining = std::chrono::seconds(5) - age;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(remaining).count();
        timeout = std::min(timeout, int(std::max<std::int64_t>(0, (us + 999) / 1000)));
      }
    }
    return timeout;
  }

public:
  explicit Server(Config config)
      : cfg(std::move(config)), backend(cfg.backend), lanes(cfg.backend.lanes, nullptr),
        flights(cfg.backend.lanes) {
    const auto socket_parent = fs::absolute(cfg.socket).parent_path();
    const auto admin_parent = fs::absolute(cfg.admin).parent_path();
    fs::create_directories(socket_parent);
    fs::create_directories(admin_parent);
    if (fs::canonical(socket_parent) == fs::canonical(admin_parent))
      throw std::runtime_error("admin token and public socket must use separate directories");
    admin = admin_token(cfg.admin);
    if (!cfg.trace.empty()) {
      auto overlaps = [](const fs::path &a, const fs::path &b) {
        if (fs::weakly_canonical(fs::absolute(a)) == fs::weakly_canonical(fs::absolute(b)))
          return true;
        std::error_code ec;
        return fs::equivalent(a, b, ec);
      };
      if (overlaps(cfg.trace, cfg.admin) || overlaps(cfg.trace, cfg.socket) ||
          fs::is_symlink(fs::symlink_status(cfg.trace)))
        throw std::runtime_error(
            "trace must not alias the admin credential or socket, or be a symlink");
      fs::create_directories(fs::absolute(cfg.trace).parent_path());
      trace.open(cfg.trace);
      if (!trace)
        throw std::runtime_error("open trace");
      chmod(cfg.trace.c_str(), 0600);
    }
    if (cfg.socket.size() >= sizeof(sockaddr_un::sun_path))
      throw std::runtime_error("socket path too long");
    listener = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (listener < 0)
      throw std::runtime_error("socket");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strcpy(address.sun_path, cfg.socket.c_str());
    if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0)
      throw std::runtime_error("bind (existing socket is never removed automatically): " +
                               std::string(strerror(errno)));
    if (chmod(cfg.socket.c_str(), 0666) < 0 || listen(listener, 128) < 0)
      throw std::runtime_error("listen");
    std::cout << object({{"event", str("ready")},
                         {"socket", str(cfg.socket)},
                         {"backend", device_json()}})
              << std::endl;
  }
  ~Server() {
    for (auto &[fd, _] : clients)
      close(fd);
    if (listener >= 0)
      close(listener);
    unlink(cfg.socket.c_str());
  }
  void run() {
    std::vector<pollfd> polls;
    polls.reserve(cfg.max_clients + 1);
    while (true) {
      if (stop_requested)
        draining = true;
      if (draining) {
        for (auto &[_, j] : jobs)
          if (!j.terminal())
            j.cancel = true;
      }
      retire();
      trim_pending_history();
      schedule();
      unsigned live = 0;
      for (const auto &[_, t] : tenants)
        live += t.live_slots;
      if (draining && live == 0) {
        bool flushed = true;
        for (auto &[_, c] : clients) {
          cancel_session(c);
          c.closing = true;
          c.input.clear();
          if (!c.close_timer_started) {
            c.close_timer_started = true;
            c.closing_since = Clock::now();
          }
          write_client(c);
          flushed = flushed && c.output.empty();
        }
        // Large responses may need multiple nonblocking sends. Keep polling
        // POLLOUT until flushed, or retire an unread peer at its close deadline.
        if (flushed)
          break;
      }
      polls.clear();
      polls.push_back({listener, short(draining ? 0 : POLLIN), 0});
      for (const auto &[fd, c] : clients)
        polls.push_back(
            {fd, short((c.closing || c.read_eof ? 0 : POLLIN) | (c.output.empty() ? 0 : POLLOUT)),
             0});
      int rc = poll(polls.data(), polls.size(), poll_timeout());
      if (rc < 0) {
        if (errno == EINTR)
          continue;
        throw std::runtime_error("poll");
      }
      if (polls[0].revents & POLLIN)
        accept_clients();
      for (std::size_t i = 1; i < polls.size(); ++i) {
        auto it = clients.find(polls[i].fd);
        if (it == clients.end())
          continue;
        auto &c = it->second;
        if ((polls[i].revents & POLLIN) || (!c.closing && c.input.find('\n') != std::string::npos))
          read_client(c);
        if (polls[i].revents & POLLOUT)
          write_client(c);
        if (polls[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
          cancel_session(c);
          c.closing = true;
          c.output.clear();
        }
      }
      std::vector<int> dead;
      for (auto &[fd, c] : clients) {
        if (!c.tenant && !c.admin_authenticated &&
            Clock::now() - c.created > std::chrono::seconds(5)) {
          c.closing = true;
        }
        if (c.closing && !c.close_timer_started) {
          c.close_timer_started = true;
          c.closing_since = Clock::now();
        }
        if (c.closing && Clock::now() - c.closing_since > std::chrono::seconds(2))
          c.output.clear();
        if (c.closing && c.output.empty())
          dead.push_back(fd);
      }
      for (int fd : dead)
        disconnect(fd);
    }
    event(object({{"event", str("shutdown")}, {"global", global_json()}}));
  }
};
static Config parse(int argc, char **argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout << "gpu-host --socket PATH --admin-file PATH [--lanes N --sm-per-lane N --slots N "
                   "--max-elements N --quantum N --batch-elements N --tick-ms N --trace PATH "
                   "--mode green|streams]\n";
      std::exit(0);
    }
    if (i + 1 >= argc)
      throw std::runtime_error("missing value for " + key);
    std::string value = argv[++i];
    if (key == "--socket")
      cfg.socket = value;
    else if (key == "--admin-file")
      cfg.admin = value;
    else if (key == "--trace")
      cfg.trace = value;
    else if (key == "--lanes")
      cfg.backend.lanes = unsigned(integer(value, 1, 8));
    else if (key == "--sm-per-lane")
      cfg.backend.sm_per_lane = unsigned(integer(value, 2, 64));
    else if (key == "--slots")
      cfg.backend.slots_per_lane = unsigned(integer(value, 1, 64));
    else if (key == "--max-elements")
      cfg.backend.max_elements_per_slot = std::size_t(integer(value, 1, 1048576));
    else if (key == "--quantum")
      cfg.quantum = std::uint32_t(integer(value, 1, 16));
    else if (key == "--batch-elements")
      cfg.batch_elements = integer(value, 1, 16777216);
    else if (key == "--tick-ms")
      cfg.tick_ms = unsigned(integer(value, 1, 100));
    else if (key == "--max-clients")
      cfg.max_clients = unsigned(integer(value, 1, 1024));
    else if (key == "--mode") {
      if (value != "green" && value != "streams")
        throw std::runtime_error("mode must be green or streams");
      cfg.backend.require_green = value == "green";
    } else
      throw std::runtime_error("unknown option " + key);
  }
  return cfg;
}
int main(int argc, char **argv) {
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  signal(SIGPIPE, SIG_IGN);
  try {
    Server server(parse(argc, argv));
    server.run();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "gpu-host: " << e.what() << '\n';
    return 1;
  }
}
