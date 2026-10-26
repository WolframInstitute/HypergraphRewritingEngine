// hgevolve._core: the engine's CPU job in-process (run_rewriting_arrays), with the reply's arrays
// handed to Python as NumPy views.
//
// A job is the paclet's envelope as Python objects: dicts, lists, ints, floats, strings, and the
// package's Symbol and Function classes (registered by register_types). It becomes a
// wxf::WXFValue and is read by the same parse_job hg_evolve uses, through wxf::ValueParser.
//
// THE ARRAYS ARE VIEWS. Every array of a Reply points into the hgffi::ArrayReply the engine
// built, and holds a capsule that owns a reference to it, so the reply lives as long as its last
// array. A Reply's arrays are read-only.
//
// ONE JOB AT A TIME PER PROCESS. The engine's session slot is per process, and a job runs on the
// engine's own workers, so calls are serialised by a mutex; the GIL is released while a job runs.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "reply_out.hpp"
#include "state_statistics.hpp"

namespace nb = nanobind;

namespace HG_NAMESPACE {
namespace ffi {
extern const char kBuildStamp[];  // hypergraph_ffi.cpp
}
}  // namespace HG_NAMESPACE

namespace {

namespace hgffi = HG_NAMESPACE::ffi;
namespace wxf = HG_NAMESPACE::wxf;
namespace stats = HG_NAMESPACE::stats;
using Owner = std::shared_ptr<const hgffi::ArrayReply>;

nb::object g_symbol_type;
nb::object g_function_type;
std::mutex g_job_mutex;

// ---------------------------------------------------------------------------------------------
// Python -> WXFValue

int64_t to_int64(PyObject* o) {
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
    if (overflow) throw nb::value_error("an integer of the job does not fit in 64 bits");
    if (v == -1 && PyErr_Occurred()) throw nb::python_error();
    return static_cast<int64_t>(v);
}

wxf::WXFValue to_value(nb::handle obj, int depth = 0) {
    if (depth > 64) throw nb::value_error("the job nests more than 64 levels");
    // The kinds an edge list is made of first: a job's bulk is lists of integers.
    PyObject* o = obj.ptr();
    if (PyLong_CheckExact(o)) return wxf::WXFValue(to_int64(o));
    if (PyList_CheckExact(o)) {
        const Py_ssize_t n = PyList_GET_SIZE(o);
        wxf::WXFValueList l;
        l.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* item = PyList_GET_ITEM(o, i);
            if (PyLong_CheckExact(item)) l.emplace_back(to_int64(item));
            else l.push_back(to_value(item, depth + 1));
        }
        return wxf::WXFValue(std::move(l));
    }
    if (obj.is_none()) return wxf::WXFValue(wxf::Symbol{"None"});
    if (PyBool_Check(obj.ptr())) return wxf::WXFValue(wxf::Symbol{obj.ptr() == Py_True ? "True" : "False"});
    if (g_symbol_type.is_valid() && nb::isinstance(obj, g_symbol_type))
        return wxf::WXFValue(wxf::Symbol{nb::cast<std::string>(nb::str(obj))});
    if (nb::isinstance<nb::str>(obj)) return wxf::WXFValue(nb::cast<std::string>(obj));
    if (g_function_type.is_valid() && nb::isinstance(obj, g_function_type)) {
        wxf::Function f;
        f.head = nb::cast<std::string>(obj.attr("head"));
        for (nb::handle a : obj.attr("args")) f.args.push_back(to_value(a, depth + 1));
        return wxf::WXFValue(std::move(f));
    }
    if (nb::isinstance<nb::dict>(obj)) {
        wxf::WXFValueAssociation a;
        for (auto [k, v] : nb::borrow<nb::dict>(obj))
            a.push_back({to_value(k, depth + 1), to_value(v, depth + 1)});
        return wxf::WXFValue(std::move(a));
    }
    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        wxf::WXFValueList l;
        for (nb::handle v : obj) l.push_back(to_value(v, depth + 1));
        return wxf::WXFValue(std::move(l));
    }
    if (nb::isinstance<nb::float_>(obj)) return wxf::WXFValue(nb::cast<double>(obj));
    if (nb::hasattr(obj, "__index__")) {
        int overflow = 0;
        nb::object i = nb::steal(PyNumber_Index(obj.ptr()));
        if (!i.is_valid()) throw nb::python_error();
        const long long v = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
        if (overflow) throw nb::value_error("an integer of the job does not fit in 64 bits");
        return wxf::WXFValue(static_cast<int64_t>(v));
    }
    if (nb::hasattr(obj, "__float__")) return wxf::WXFValue(nb::cast<double>(obj.attr("__float__")()));
    throw nb::type_error(("a job cannot hold a value of type " +
                          nb::cast<std::string>(nb::str(obj.type()))).c_str());
}

// ---------------------------------------------------------------------------------------------
// WXFValue -> Python (the reply's small values: counts, statistics, warnings, ...)

nb::object from_value(const wxf::WXFValue& v);

// An association key: a list key (a histogram's arity list) as a tuple, which is hashable.
nb::object from_key(const wxf::WXFValue& k) {
    if (!k.holds<wxf::WXFValueList>()) return from_value(k);
    const auto& items = k.get<wxf::WXFValueList>();
    nb::list l;
    for (const auto& item : items) l.append(from_key(item));
    return nb::tuple(l);
}

nb::object real_array(const wxf::PackedReals& p) {
    std::vector<size_t> shape(p.dims.begin(), p.dims.end());
    auto* copy = new std::vector<double>(p.values);
    nb::capsule owner(copy, [](void* q) noexcept { delete static_cast<std::vector<double>*>(q); });
    return nb::cast(nb::ndarray<nb::numpy, double>(copy->data(), shape.size(), shape.data(), owner));
}

nb::object from_value(const wxf::WXFValue& v) {
    return std::visit(
        [](const auto& x) -> nb::object {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return nb::list();
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return nb::int_(x);
            } else if constexpr (std::is_same_v<T, double>) {
                return nb::float_(x);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return nb::str(x.c_str(), x.size());
            } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                return nb::bytes(reinterpret_cast<const char*>(x.data()), x.size());
            } else if constexpr (std::is_same_v<T, wxf::WXFValueList>) {
                nb::list l;
                for (const auto& item : x) l.append(from_value(item));
                return l;
            } else if constexpr (std::is_same_v<T, wxf::WXFValueAssociation>) {
                nb::dict d;
                for (const auto& [k, item] : x) d[from_key(k)] = from_value(item);
                return d;
            } else if constexpr (std::is_same_v<T, wxf::Symbol>) {
                if (x.name == "True") return nb::bool_(true);
                if (x.name == "False") return nb::bool_(false);
                if (x.name == "None") return nb::none();
                return nb::str(x.name.c_str(), x.name.size());
            } else if constexpr (std::is_same_v<T, wxf::Boxed<wxf::Function>>) {
                nb::list args;
                for (const auto& item : x->args) args.append(from_value(item));
                return nb::make_tuple(nb::str(x->head.c_str(), x->head.size()), args);
            } else if constexpr (std::is_same_v<T, wxf::Boxed<wxf::PackedReals>>) {
                return real_array(*x);
            } else {
                throw std::runtime_error("the reply holds a value encoded as WXF bytes");
            }
        },
        v.data);
}

// ---------------------------------------------------------------------------------------------
// Views into the reply

template <class T>
nb::object view(const Owner& owner, const T* data, size_t rows, size_t cols = 0) {
    static const T empty{};
    auto* keep = new Owner(owner);
    nb::capsule cap(keep, [](void* p) noexcept { delete static_cast<Owner*>(p); });
    const size_t shape[2] = {rows, cols};
    return nb::cast(nb::ndarray<nb::numpy, const T>(rows ? data : &empty, cols ? 2 : 1, shape, cap));
}

template <class T>
nb::object view(const Owner& owner, const std::vector<T>& v) {
    return view(owner, v.data(), v.size());
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

struct Reply {
    Owner r;

    // {table: (rows, [(column, kind, {part: array})])}
    nb::dict tables() const {
        nb::dict out;
        for (const auto& [name, t] : r->tables) {
            nb::list cols;
            for (const auto& [cname, c] : t.columns) {
                nb::dict parts;
                parts["values"] = view(r, c.ints);
                if (c.kind != hgffi::Column::Kind::Int) parts["offsets"] = view(r, c.offsets);
                if (c.kind == hgffi::Column::Kind::Edges) {
                    parts["vertex_offsets"] = view(r, c.vertex_offsets);
                    parts["vertices"] = view(r, c.vertices);
                }
                if (c.kind == hgffi::Column::Kind::VertexReals) parts["reals"] = view(r, c.reals);
                cols.append(nb::make_tuple(nb::str(cname.c_str()), nb::str(kind_name(c.kind)), parts));
            }
            out[nb::str(name.c_str())] = nb::make_tuple(t.rows, cols);
        }
        return out;
    }

    // {relation: (names, rows x len(names) array)}
    nb::dict relations() const {
        nb::dict out;
        for (const auto& [name, rel] : r->relations) {
            nb::list names;
            for (const auto& n : rel.names) names.append(nb::str(n.c_str()));
            out[nb::str(name.c_str())] =
                nb::make_tuple(names, view(r, rel.cells.data(), rel.rows, rel.names.size()));
        }
        return out;
    }

    nb::dict values() const {
        nb::dict out;
        for (const auto& [k, v] : r->values) out[from_key(k)] = from_value(v);
        return out;
    }

    // The address of a table column's part in the reply: the data pointer its view must have.
    uintptr_t address(const std::string& table, const std::string& column,
                      const std::string& part) const {
        for (const auto& [name, t] : r->tables) {
            if (name != table) continue;
            for (const auto& [cname, c] : t.columns) {
                if (cname != column) continue;
                if (part == "values") return reinterpret_cast<uintptr_t>(c.ints.data());
                if (part == "offsets") return reinterpret_cast<uintptr_t>(c.offsets.data());
                if (part == "vertex_offsets") return reinterpret_cast<uintptr_t>(c.vertex_offsets.data());
                if (part == "vertices") return reinterpret_cast<uintptr_t>(c.vertices.data());
                if (part == "reals") return reinterpret_cast<uintptr_t>(c.reals.data());
            }
        }
        throw nb::key_error((table + "/" + column + "/" + part).c_str());
    }
};

Reply run(nb::handle job) {
    const wxf::WXFValue value = to_value(job);
    HostBridge host;
    host.progress = [](const std::string& line) {
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
    };
    std::shared_ptr<hgffi::ArrayReply> reply;
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(g_job_mutex);
        reply = std::make_shared<hgffi::ArrayReply>(run_rewriting_arrays(value, host));
    }
    return Reply{std::move(reply)};
}

nb::bytes encode(nb::handle job) {
    const wxf::WXFValue value = to_value(job);
    wxf::Writer w;
    w.write_header();
    w.write(value);
    return nb::bytes(reinterpret_cast<const char*>(w.data().data()), w.size());
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "The engine's CPU job in-process; see the hgevolve package.";
    m.def("register_types", [](nb::object symbol, nb::object function) {
        g_symbol_type = std::move(symbol);
        g_function_type = std::move(function);
    });
    m.def("run", &run, nb::arg("job"),
          "Run a job (the paclet's envelope as Python objects) and return its Reply.");
    m.def("encode", &encode, nb::arg("job"), "The job's WXF bytes, as hg_evolve reads them.");
    m.def("build_stamp", []() { return std::string(hgffi::kBuildStamp); });
    m.def("measure_names", []() {
        nb::list l;
        for (const auto& mn : stats::kMeasureNames) l.append(nb::str(mn.name));
        return l;
    });
    m.def("matrix_names", []() {
        nb::list l;
        for (const char* n : stats::kMatrixNames) l.append(nb::str(n));
        return l;
    });
    nb::class_<Reply>(m, "Reply")
        .def("tables", &Reply::tables)
        .def("relations", &Reply::relations)
        .def("values", &Reply::values)
        .def("address", &Reply::address);
    // Types registered with nanobind outlive the module's globals only if released first.
    nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
        g_symbol_type = nb::object();
        g_function_type = nb::object();
    }));
}
