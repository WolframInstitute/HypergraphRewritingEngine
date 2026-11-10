// The engine's CPU job for JavaScript (Emscripten embind): the job code of the paclet and of the
// Python module (run_rewriting_arrays), with the reply copied into JavaScript objects.
//
// A job is the paclet's envelope as JavaScript values. Numbers that are integers become WXF
// integers and other numbers reals; a BigInt is an integer; true, false and null are the
// symbols True, False and None; {$symbol: "Full"} is a symbol and {$head: "Rule", args: [...]}
// is head[args...]; arrays are lists and other objects are associations. js/hgevolve.mjs builds
// the envelope from evolve()'s arguments.
//
// THE ARRAYS ARE COPIES. A typed array over the module's heap is invalidated when the heap grows,
// so every array of a reply is copied out of the heap before run() returns.
//
// ONE JOB AT A TIME. The engine's session slot is per module instance, and run() is synchronous.
// In the threaded build the job runs on the engine's worker threads (Web Workers), and a
// transition cost callback is refused: a JavaScript function is callable only from the thread
// that made it.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "ffi_job.hpp"   // set_transition_cost_hook
#include "hg_core.hpp"
#include "hgcommon/park.hpp"
#include "hgcommon/transition_cost_core.hpp"
#include "reply_out.hpp"
#include "state_statistics.hpp"

namespace HG_NAMESPACE {
namespace ffi {
extern const char kBuildStamp[];  // hypergraph_ffi.cpp
}
}  // namespace HG_NAMESPACE

namespace {

using emscripten::val;
namespace hgffi = HG_NAMESPACE::ffi;
namespace wxf = HG_NAMESPACE::wxf;
namespace stats = HG_NAMESPACE::stats;
namespace hc = HG_NAMESPACE::common;

[[noreturn]] void type_error(const std::string& message) {
    val::global("TypeError").new_(val(message)).throw_();
}

// ---------------------------------------------------------------------------------------------
// JavaScript -> WXFValue

wxf::WXFValue to_value(const val& v, int depth = 0) {
    if (depth > 64) type_error("the job nests more than 64 levels");
    const std::string type = v.typeOf().as<std::string>();
    if (type == "number") {
        const double d = v.as<double>();
        // Integers up to 2^53 are exact in a JavaScript number; larger ones are given as BigInt.
        if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) <= 9007199254740992.0)
            return wxf::WXFValue(static_cast<int64_t>(d));
        return wxf::WXFValue(d);
    }
    if (type == "bigint") {
        const val lo = val::global("BigInt").call<val>("asIntN", 64, v);
        if (!lo.strictlyEquals(v)) type_error("an integer of the job does not fit in 64 bits");
        return wxf::WXFValue(static_cast<int64_t>(lo.as<int64_t>()));
    }
    if (type == "string") return wxf::WXFValue(v.as<std::string>());
    if (type == "boolean") return wxf::WXFValue(wxf::Symbol{v.as<bool>() ? "True" : "False"});
    if (v.isNull() || v.isUndefined()) return wxf::WXFValue(wxf::Symbol{"None"});
    if (type != "object") type_error("a job cannot hold a value of type " + type);
    if (val::global("Array").call<bool>("isArray", v) ||
        val::global("ArrayBuffer").call<bool>("isView", v)) {
        const unsigned n = v["length"].as<unsigned>();
        wxf::WXFValueList l;
        l.reserve(n);
        for (unsigned i = 0; i < n; ++i) l.push_back(to_value(v[i], depth + 1));
        return wxf::WXFValue(std::move(l));
    }
    if (v.hasOwnProperty("$symbol")) return wxf::WXFValue(wxf::Symbol{v["$symbol"].as<std::string>()});
    if (v.hasOwnProperty("$head")) {
        wxf::Function f;
        f.head = v["$head"].as<std::string>();
        const val args = v["args"];
        const unsigned n = args.isUndefined() ? 0 : args["length"].as<unsigned>();
        for (unsigned i = 0; i < n; ++i) f.args.push_back(to_value(args[i], depth + 1));
        return wxf::WXFValue(std::move(f));
    }
    const val keys = val::global("Object").call<val>("keys", v);
    const unsigned n = keys["length"].as<unsigned>();
    wxf::WXFValueAssociation a;
    a.reserve(n);
    for (unsigned i = 0; i < n; ++i) {
        const val k = keys[i];
        a.push_back({wxf::WXFValue(k.as<std::string>()), to_value(v[k], depth + 1)});
    }
    return wxf::WXFValue(std::move(a));
}

// ---------------------------------------------------------------------------------------------
// WXFValue -> JavaScript (the reply's values: counts, statistics, warnings, ...)

val from_value(const wxf::WXFValue& v);

// A reply integer: a number when it is exact as one, a BigInt otherwise (canonical hashes).
val integer(int64_t x) {
    if (x >= -9007199254740991ll && x <= 9007199254740991ll) return val(static_cast<double>(x));
    return val(x);
}

// An association key as a property name: a list key (a histogram's arity list) as its
// comma-joined values.
std::string key_name(const wxf::WXFValue& k) {
    if (k.holds<std::string>()) return k.get<std::string>();
    if (k.holds<wxf::Symbol>()) return k.get<wxf::Symbol>().name;
    if (k.holds<int64_t>()) return std::to_string(k.get<int64_t>());
    if (k.holds<wxf::WXFValueList>()) {
        std::string s;
        for (const auto& item : k.get<wxf::WXFValueList>()) {
            if (!s.empty()) s += ",";
            s += key_name(item);
        }
        return s;
    }
    return from_value(k).call<std::string>("toString");
}

val real_array(const wxf::PackedReals& p) {
    val out = val::object();
    out.set("dims", val::array(std::vector<double>(p.dims.begin(), p.dims.end())));
    out.set("values", val(emscripten::typed_memory_view(p.values.size(), p.values.data()))
                          .call<val>("slice"));
    return out;
}

val from_value(const wxf::WXFValue& v) {
    return std::visit(
        [](const auto& x) -> val {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return val::array();
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return integer(x);
            } else if constexpr (std::is_same_v<T, double>) {
                return val(x);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return val(x);
            } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                return val(emscripten::typed_memory_view(x.size(), x.data())).call<val>("slice");
            } else if constexpr (std::is_same_v<T, wxf::WXFValueList>) {
                val l = val::array();
                for (const auto& item : x) l.call<void>("push", from_value(item));
                return l;
            } else if constexpr (std::is_same_v<T, wxf::WXFValueAssociation>) {
                val o = val::object();
                for (const auto& [k, item] : x) o.set(key_name(k), from_value(item));
                return o;
            } else if constexpr (std::is_same_v<T, wxf::Symbol>) {
                if (x.name == "True") return val(true);
                if (x.name == "False") return val(false);
                if (x.name == "None") return val::null();
                return val(x.name);
            } else if constexpr (std::is_same_v<T, wxf::Boxed<wxf::Function>>) {
                val args = val::array();
                for (const auto& item : x->args) args.call<void>("push", from_value(item));
                val f = val::object();
                f.set("$head", val(x->head));
                f.set("args", args);
                return f;
            } else if constexpr (std::is_same_v<T, wxf::Boxed<wxf::PackedReals>>) {
                return real_array(*x);
            } else {
                type_error("the reply holds a value encoded as WXF bytes");
            }
        },
        v.data);
}

// ---------------------------------------------------------------------------------------------
// The reply's arrays, copied out of the heap

template <class T>
val copy(const std::vector<T>& v) {
    return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

const char* kind_name(hgffi::Column::Kind k) {
    switch (k) {
        case hgffi::Column::Kind::Int: return "int";
        case hgffi::Column::Kind::IntLists: return "lists";
        case hgffi::Column::Kind::Edges: return "edges";
        case hgffi::Column::Kind::VertexReals: return "vertex_reals";
    }
    return "?";
}

// {values, tables: {name: {rows, columns: [{name, kind, values, offsets, ...}]}},
//  relations: {name: {names, rows, cells}}}. Integer arrays are BigInt64Array, reals
// Float64Array.
val reply_object(const hgffi::ArrayReply& r) {
    val tables = val::object();
    for (const auto& [name, t] : r.tables) {
        val cols = val::array();
        for (const auto& [cname, c] : t.columns) {
            val col = val::object();
            col.set("name", val(cname));
            col.set("kind", val(kind_name(c.kind)));
            col.set("values", copy(c.ints));
            if (c.kind != hgffi::Column::Kind::Int) col.set("offsets", copy(c.offsets));
            if (c.kind == hgffi::Column::Kind::Edges) {
                col.set("vertexOffsets", copy(c.vertex_offsets));
                col.set("vertices", copy(c.vertices));
            }
            if (c.kind == hgffi::Column::Kind::VertexReals) col.set("reals", copy(c.reals));
            cols.call<void>("push", col);
        }
        val table = val::object();
        table.set("rows", val(static_cast<double>(t.rows)));
        table.set("columns", cols);
        tables.set(name, table);
    }
    val relations = val::object();
    for (const auto& [name, rel] : r.relations) {
        val names = val::array();
        for (const auto& n : rel.names) names.call<void>("push", val(n));
        val relation = val::object();
        relation.set("names", names);
        relation.set("rows", val(static_cast<double>(rel.rows)));
        relation.set("cells", copy(rel.cells));
        relations.set(name, relation);
    }
    val values = val::object();
    for (const auto& [k, v] : r.values) values.set(key_name(k), from_value(v));
    val out = val::object();
    out.set("values", values);
    out.set("tables", tables);
    out.set("relations", relations);
    return out;
}

// A state's hyperedges as an array of arrays of vertex ids.
val edge_lists(const hc::TcEdges& e) {
    val out = val::array();
    for (uint32_t i = 0; i < e.count(); ++i) {
        val edge = val::array();
        for (uint32_t k = 0; k < e.arity(i); ++k) edge.call<void>("push", val(e.at(i, k)));
        out.call<void>("push", edge);
    }
    return out;
}

// The first error a transition cost callback throws ends the job's use of the callback: its
// transition and every later one cost +Infinity, and run() throws the error after the job.
struct CostCallback {
    val fn = val::undefined();
    val error = val::undefined();
    bool failed = false;

    double operator()(const hc::TcEdges& parent, const hc::TcEdges& child) {
        if (failed) return HUGE_VAL;
        const val wrapped = val::module_property("hgCallCost")(fn, edge_lists(parent),
                                                               edge_lists(child));
        if (wrapped["ok"].as<bool>()) return wrapped["value"].as<double>();
        error = wrapped["error"];
        failed = true;
        return HUGE_VAL;
    }
};

val run(const val& job, const val& transition_cost) {
    const wxf::WXFValue value = to_value(job);
    // An option the engine ignores (unknown, or a value it cannot read) refuses the job before it
    // runs, as HGEvolve and the Python package refuse it.
    std::vector<std::string> problems;
    try {
        problems = job_option_problems(value);
    } catch (const std::exception& e) {
        val::module_property("HgEngineError").new_(val(std::string(e.what()))).throw_();
    }
    if (!problems.empty()) {
        std::string message;
        for (const auto& p : problems) message += (message.empty() ? "" : "; ") + p;
        val::module_property("HgEngineError").new_(val(message)).throw_();
    }
    HostBridge host;
    host.progress = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
    CostCallback callback;
    const bool has_callback = !transition_cost.isUndefined() && !transition_cost.isNull();
    if (has_callback) {
        if (hc::build_has_threads)
            type_error("a transition cost function needs the single-threaded build: a JavaScript "
                       "function is callable only from its own thread");
        callback.fn = transition_cost;
        hgffi::set_transition_cost_hook(
            [&callback](const hc::TcEdges& p, const hc::TcEdges& c) { return callback(p, c); });
    }
    std::unique_ptr<hgffi::ArrayReply> reply;
    std::string failure;
    try {
        reply = std::make_unique<hgffi::ArrayReply>(run_rewriting_arrays(value, host));
    } catch (const std::exception& e) {
        failure = e.what();
        if (failure.empty()) failure = "the engine refused the job";
    }
    hgffi::set_transition_cost_hook(nullptr);
    release_job_memory(0);
    if (callback.failed) callback.error.throw_();
    if (!reply) val::module_property("HgEngineError").new_(val(failure)).throw_();
    return reply_object(*reply);
}

val encode(const val& job) {
    const wxf::WXFValue value = to_value(job);
    wxf::Writer w;
    w.write_header();
    w.write(value);
    return val(emscripten::typed_memory_view(w.size(), w.data().data())).call<val>("slice");
}

// hgcommon::tc_transition_cost, the engine's heat residual, on two edge lists.
double heat_residual(const val& parent, const val& child, const std::string& field,
                     double time_step, const std::string& graph) {
    hc::TransitionCostSpec spec;
    if (field == "Degree") spec.field = hc::TC_FIELD_DEGREE;
    else if (field == "Curvature") spec.field = hc::TC_FIELD_CURVATURE;
    else if (field == "LoopCount") spec.field = hc::TC_FIELD_LOOP_COUNT;
    else if (field == "EdgeMultiplicity") spec.field = hc::TC_FIELD_EDGE_MULTIPLICITY;
    else type_error("field is one of 'Degree', 'Curvature', 'LoopCount' and 'EdgeMultiplicity', "
                    "got '" + field + "'");
    if (graph == "Consecutive") spec.graph = hc::SG_GRAPH_CONSECUTIVE;
    else if (graph == "AllPairs") spec.graph = hc::SG_GRAPH_ALL_PAIRS;
    else type_error("statisticsGraph is 'Consecutive' or 'AllPairs', got '" + graph + "'");
    spec.dt = time_step;
    hc::TcEdgeLists p, c;
    auto fill = [](const val& edges, hc::TcEdgeLists& out) {
        std::vector<uint32_t> verts;
        const unsigned n = edges["length"].as<unsigned>();
        for (unsigned i = 0; i < n; ++i) {
            const val e = edges[i];
            const unsigned m = e["length"].as<unsigned>();
            verts.clear();
            for (unsigned k = 0; k < m; ++k) {
                const double x = e[k].as<double>();
                if (!(x >= 0 && x <= 4294967294.0) || x != std::floor(x))
                    type_error("a vertex id is an integer from 0 to 2^32 - 2");
                verts.push_back(static_cast<uint32_t>(x));
            }
            out.add(verts.data(), static_cast<uint32_t>(verts.size()));
        }
    };
    fill(parent, p);
    fill(child, c);
    return hc::tc_transition_cost(p.view(), c.view(), spec);
}

val measure_names() {
    val l = val::array();
    for (const auto& mn : stats::kMeasureNames) l.call<void>("push", val(mn.name));
    return l;
}

val matrix_names() {
    val l = val::array();
    for (const char* n : stats::kMatrixNames) l.call<void>("push", val(n));
    return l;
}

std::string build_stamp() { return std::string(hgffi::kBuildStamp); }

bool threaded() { return hc::build_has_threads; }

}  // namespace

EMSCRIPTEN_BINDINGS(hgevolve) {
    emscripten::function("runJob", &run);
    emscripten::function("encode", &encode);
    emscripten::function("heatResidual", &heat_residual);
    emscripten::function("measureNames", &measure_names);
    emscripten::function("matrixNames", &matrix_names);
    emscripten::function("buildStamp", &build_stamp);
    emscripten::function("threaded", &threaded);
}
