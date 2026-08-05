// The option surface is written down in three places, and they must agree.
//
// A user-facing option is declared in Options[HGEvolve], SENT by the WL wrapper in the options
// association, PARSED by the FFI, and described in the reference page. Nothing links those four
// copies, so an option can be declared and never sent, sent and never parsed, or documented
// after it stops existing -- each of which reads as a working option and does nothing.
//
// That is not hypothetical: RandomSeed was declared, documented "for reproducibility", consumed
// by the initial-condition generators, and never sent -- so a sampled evolution ignored it. This
// gate is the check that found it, made standing.
//
// It reads the sources rather than any generated artifact, because the point is that the copies
// agree with EACH OTHER; comparing two views of one generated list would prove nothing.

#include <gtest/gtest.h>

#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

// A repository file, addressed from the source tree CMake configured rather than from the working
// directory. Guessing a prefix off the caller's cwd finds the file from some directories and not
// others, and an empty result here reads as "the option is not declared anywhere" -- a missing
// file would make this gate pass by finding no options to disagree about.
std::string read_repo_file(const std::string& rel) {
    std::ifstream in(std::string(HG_SOURCE_DIR) + "/" + rel);
    if (!in) return {};
    return std::string((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
}

// Names captured by `re` inside the region between `begin` and `end` markers. An empty region
// yields nothing, which every caller asserts against -- a regex that silently stops matching
// would otherwise turn this gate into a tautology.
std::set<std::string> names_in_region(const std::string& text, const std::string& begin,
                                      const std::string& end, const std::regex& re) {
    std::set<std::string> out;
    const size_t b = text.find(begin);
    if (b == std::string::npos) return out;
    const size_t e = text.find(end, b + begin.size());
    if (e == std::string::npos) return out;
    const std::string region = text.substr(b, e - b);
    for (auto it = std::sregex_iterator(region.begin(), region.end(), re);
         it != std::sregex_iterator(); ++it) {
        out.insert((*it)[1].str());
    }
    return out;
}

}  // namespace

TEST(OptionSurface, EveryOptionTheWrapperSendsIsParsedByTheFfi) {
    const std::string wl  = read_repo_file("paclet/Kernel/HypergraphRewriting.wl");
    const std::string ffi = read_repo_file("paclet_source/hypergraph_ffi.cpp");
    ASSERT_FALSE(wl.empty()) << "paclet/Kernel/HypergraphRewriting.wl not found";
    ASSERT_FALSE(ffi.empty()) << "paclet_source/hypergraph_ffi.cpp not found";

    const std::set<std::string> sent =
        names_in_region(wl, "  options = <|", "  |>;", std::regex("\"([A-Za-z]+)\"\\s*->"));
    std::set<std::string> parsed;
    {
        const std::regex re("option_key == \"([A-Za-z]+)\"");
        for (auto it = std::sregex_iterator(ffi.begin(), ffi.end(), re);
             it != std::sregex_iterator(); ++it) {
            parsed.insert((*it)[1].str());
        }
    }
    ASSERT_FALSE(sent.empty()) << "found no options association in the wrapper; the regex has "
                                  "stopped matching and this gate is asserting nothing";
    ASSERT_FALSE(parsed.empty()) << "found no option keys in the FFI parser; same";

    for (const std::string& key : sent) {
        EXPECT_TRUE(parsed.count(key))
            << "the wrapper sends \"" << key << "\" and the FFI does not parse it, so setting it "
            << "does nothing and says nothing";
    }
}

TEST(OptionSurface, EveryDocumentedOptionIsAnOptionHGEvolveAccepts) {
    const std::string wl  = read_repo_file("paclet/Kernel/HypergraphRewriting.wl");
    const std::string doc = read_repo_file("paclet/Documentation/Source/HGEvolve.md");
    ASSERT_FALSE(wl.empty()) << "paclet/Kernel/HypergraphRewriting.wl not found";
    ASSERT_FALSE(doc.empty()) << "paclet/Documentation/Source/HGEvolve.md not found";

    const std::set<std::string> declared =
        names_in_region(wl, "Options[HGEvolve] = {", "\n};", std::regex("\"([A-Za-z]+)\"\\s*->"));
    // MOST OPTIONS ARE DOCUMENTED IN A TABLE, NOT UNDER A HEADING. Matching only
    // `### "Name"` checked 10 of the 78 names the page carried, and the 68 it skipped
    // included 41 for analyses that moved to the companion project -- documented, accepted
    // by nothing, and green here throughout. So both shapes are collected: the heading, and
    // the FIRST CELL of a table row, which may name several options at once
    // (`| "GridWidth", "GridHeight" | 10, 10 | ... |`).
    std::set<std::string> documented;
    {
        const std::regex heading("### \"([A-Za-z]+)\"");
        for (auto it = std::sregex_iterator(doc.begin(), doc.end(), heading);
             it != std::sregex_iterator(); ++it) {
            documented.insert((*it)[1].str());
        }
        // ONLY TABLES THAT HAVE A `Default` COLUMN. The page also carries a table of
        // PROPERTIES -- "StatesGraph", "CausalEdges", "NumStates" -- which are the third
        // argument, not options, and are correctly absent from Options[HGEvolve]. A scan
        // that took every table would report all of them as undeclared options.
        const std::regex quoted("\"([A-Za-z]+)\"");
        std::istringstream lines(doc);
        bool in_option_table = false;
        for (std::string line; std::getline(lines, line);) {
            if (line.empty() || line[0] != '|') {
                in_option_table = false;     // any non-row ends the table
                continue;
            }
            if (line.find("Default") != std::string::npos) {
                in_option_table = true;      // this is the header row
                continue;
            }
            if (!in_option_table) continue;
            const size_t cell_end = line.find('|', 1);
            if (cell_end == std::string::npos) continue;
            const std::string cell = line.substr(1, cell_end - 1);
            for (auto it = std::sregex_iterator(cell.begin(), cell.end(), quoted);
                 it != std::sregex_iterator(); ++it) {
                documented.insert((*it)[1].str());
            }
        }
    }
    ASSERT_FALSE(declared.empty()) << "found no Options[HGEvolve] list; the regex has stopped "
                                      "matching and this gate is asserting nothing";
    ASSERT_FALSE(documented.empty()) << "found no documented options; same";

    for (const std::string& key : documented) {
        EXPECT_TRUE(declared.count(key))
            << "the reference page documents \"" << key << "\", which HGEvolve does not accept";
    }
}
