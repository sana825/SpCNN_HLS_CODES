#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <dirent.h>
#include <sys/stat.h>
#include "inference.h"

#ifndef TEST_DATA_ROOT
#define TEST_DATA_ROOT "output_headers_9"
#endif

// ============================================================
// Class labels
// ============================================================

const char* CLASS_LABELS[94] = {
    "!",  "#", "$", "%", "&", "\'", "(", ")", "+", ",", "-", "0", "1", "2",
    "3",  "4", "5", "6", "7", "8",  "9", ";", "@", "A", "B", "C", "D", "E",
    "F",  "G", "H", "I", "J", "K",  "L", "M", "N", "O", "P", "Q", "R", "S",
    "T",  "U", "V", "W", "X", "Y",  "Z", "[", "]", "^", "_", "`", "a", "b",
    "\\", "c", ":", "d", ".", "\"", "e", "=", "f", "g", ">", "h", "i", "j",
    "k",  "l", "<", "m", "n", "o",  "|", "p", "q", "?", "r", "s", "/", "*",
    "t",  "u", "v", "w", "x", "y",  "z", "{", "}", "~"
};

// ============================================================
// Hardcoded folder -> class index table (94 entries, exact order)
// ============================================================

struct FolderEntry {
    const char* folder_name;
    int         class_idx;
};

static const FolderEntry FOLDERS[94] = {
    { "!",       0  }, { "#",       1  }, { "$",       2  }, { "%",       3  },
    { "&",       4  }, { "'",       5  }, { "(",       6  }, { ")",       7  },
    { "+",       8  }, { ",",       9  }, { "-",       10 }, { "0",       11 },
    { "1",       12 }, { "2",       13 }, { "3",       14 }, { "4",       15 },
    { "5",       16 }, { "6",       17 }, { "7",       18 }, { "8",       19 },
    { "9",       20 }, { ";",       21 }, { "@",       22 },
    { "CA",      23 }, { "CB",      24 }, { "CC",      25 }, { "CD",      26 },
    { "CE",      27 }, { "CF",      28 }, { "CG",      29 }, { "CH",      30 },
    { "CI",      31 }, { "CJ",      32 }, { "CK",      33 }, { "CL",      34 },
    { "CM",      35 }, { "CN",      36 }, { "CO",      37 }, { "CP",      38 },
    { "CQ",      39 }, { "CR",      40 }, { "CS",      41 }, { "CT",      42 },
    { "CU",      43 }, { "CV",      44 }, { "CW",      45 }, { "CX",      46 },
    { "CY",      47 }, { "CZ",      48 },
    { "[",       49 }, { "]",       50 }, { "^",       51 }, { "_",       52 },
    { "`",       53 }, { "a",       54 }, { "b",       55 }, { "bslash",  56 },
    { "c",       57 }, { "colon",   58 }, { "d",       59 }, { "dot",     60 },
    { "dqts",    61 }, { "e",       62 }, { "equal",   63 }, { "f",       64 },
    { "g",       65 }, { "greater", 66 }, { "h",       67 }, { "i",       68 },
    { "j",       69 }, { "k",       70 }, { "l",       71 }, { "less",    72 },
    { "m",       73 }, { "n",       74 }, { "o",       75 }, { "or",      76 },
    { "p",       77 }, { "q",       78 }, { "qmark",   79 }, { "r",       80 },
    { "s",       81 }, { "slash",   82 }, { "star",    83 }, { "t",       84 },
    { "u",       85 }, { "v",       86 }, { "w",       87 }, { "x",       88 },
    { "y",       89 }, { "z",       90 }, { "{",       91 }, { "}",       92 },
    { "~",       93 }
};

// ============================================================
// Parse fixed_t array from a .h file
// ============================================================

bool parse_h_file(const std::string& path, fixed_t pixels[4096]) {
    std::ifstream f(path.c_str());
    if (!f.is_open()) return false;

    std::ostringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    size_t open  = content.find('{');
    size_t close = content.find('}', open);
    if (open == std::string::npos || close == std::string::npos)
        return false;

    std::string data = content.substr(open + 1, close - open - 1);

    int idx = 0;
    std::istringstream token_stream(data);
    std::string token;
    while (std::getline(token_stream, token, ',') && idx < 4096) {
        size_t start = token.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = token.find_last_not_of(" \t\r\nf");
        if (end == std::string::npos) continue;
        std::string val_str = token.substr(start, end - start + 1);
        if (val_str.empty()) continue;
        // atof returns double; single explicit cast to fixed_t, no
        // ambiguity since this is construction, not a mixed operator.
        double v = atof(val_str.c_str());
        pixels[idx++] = (fixed_t)v;
    }
    return (idx == 4096);
}

// ============================================================
// List all .h files in a directory (sorted)
// ============================================================

std::vector<std::string> list_h_files(const std::string& path) {
    std::vector<std::string> files;
    DIR* dir = opendir(path.c_str());
    if (!dir) return files;
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        std::string name = ent->d_name;
        if (name.size() > 2 && name.substr(name.size() - 2) == ".h")
            files.push_back(name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================
// Helpers
// ============================================================

void copy_from_flat_array(fixed_t* src,
                          fixed_t dest[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    for (int i = 0; i < IMAGE_HEIGHT; i++)
        for (int j = 0; j < IMAGE_WIDTH; j++)
            dest[i][j] = src[i * IMAGE_WIDTH + j];
}

void preprocess(fixed_t in[IMAGE_HEIGHT][IMAGE_WIDTH],
                fixed_t out[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    for (int i = 0; i < IMAGE_HEIGHT; i++)
        for (int j = 0; j < IMAGE_WIDTH; j++)
            out[i][j] = in[i][j];
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {

    const char* root = (argc > 1) ? argv[1] : TEST_DATA_ROOT;

    std::cout << "============================================================\n"
              << "  SPECTRAL CNN -- BATCH ACCURACY EVALUATION\n"
              << "  94-CLASS CHARACTER RECOGNITION (fixed-point)\n"
              << "============================================================\n\n";

    std::cout << "Mode         : FULL TEST (all samples)\n";
    std::cout << "Dataset root : " << root << "\n\n";

    int class_total  [94] = {0};
    int class_correct[94] = {0};
    int global_total   = 0;
    int global_correct = 0;
    int skipped        = 0;

    fixed_t pixels[4096];
    fixed_t img  [IMAGE_HEIGHT][IMAGE_WIDTH];
    fixed_t proc [IMAGE_HEIGHT][IMAGE_WIDTH];

    // Logits are fixed_t for fixed-point inference
    fixed_t logits[NUM_CLASSES];

    std::clock_t batch_start = std::clock();

    // ---- Loop over the 94 hardcoded folders ----
    for (int fi = 0; fi < 94; fi++) {

        const char* folder_name = FOLDERS[fi].folder_name;
        int         true_class  = FOLDERS[fi].class_idx;

        std::string folder_path = std::string(root) + "/" + folder_name;

        struct stat st;
        if (stat(folder_path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            std::cerr << "  [WARN] Folder not found: " << folder_path << "\n";
            continue;
        }

        std::vector<std::string> files = list_h_files(folder_path);
        if (files.empty()) {
            std::cerr << "  [WARN] No .h files in: " << folder_path << "\n";
            continue;
        }

        // ---- Loop over .h files in this folder ----
        for (size_t hi = 0; hi < files.size(); hi++) {

            std::string fpath = folder_path + "/" + files[hi];

            if (!parse_h_file(fpath, pixels)) {
                std::cerr << "  [WARN] Failed to parse: " << fpath << "\n";
                skipped++;
                continue;
            }

            copy_from_flat_array(pixels, img);
            preprocess(img, proc);

            // spcnn_inference: fixed_t[H][W] -> fixed_t[NUM_CLASSES]
            spcnn_inference(proc, logits);

            // get_prediction: fixed_t[NUM_CLASSES] -> int
            int  pred_class = get_prediction(logits);
            bool correct    = (pred_class == true_class);

            class_total  [true_class]++;
            if (correct) class_correct[true_class]++;
            global_total++;
            if (correct) global_correct++;

            if (global_total % 100 == 0) {
                // Pure double arithmetic here -- global_correct/global_total
                // are plain ints, so there is no fixed_t operand and no
                // ambiguity. Keep this as double throughout for reporting.
                double racc = 100.0 * global_correct / global_total;
                std::cout << "  [" << std::setw(5) << global_total << "]"
                          << "  Folder: " << std::setw(10) << folder_name
                          << "  Running accuracy: "
                          << std::fixed << std::setprecision(2) << racc << "%\n";
            }
        }
    }

    std::clock_t batch_end = std::clock();
    double total_ms = 1000.0 * (batch_end - batch_start) / CLOCKS_PER_SEC;

    // ---- Per-class accuracy report ----
    std::cout << "\n============================================================\n"
              << "  PER-CLASS ACCURACY REPORT\n"
              << "============================================================\n";
    std::cout << std::left
              << std::setw(6)  << "Idx"
              << std::setw(10) << "Label"
              << std::setw(12) << "Folder"
              << std::setw(9)  << "Correct"
              << std::setw(8)  << "Total"
              << "Accuracy\n"
              << std::string(52, '-') << "\n";

    // All accuracy bookkeeping below uses plain double, computed from the
    // int counters. fixed_t is never mixed with a float/double literal or
    // variable in a comparison / arithmetic operator, which is what caused
    // the ambiguous-overload errors in the original version.
    double sum_class_acc  = 0.0;
    int    classes_tested = 0;
    int    worst_class = -1, best_class = -1;
    double worst_acc = 200.0, best_acc = -1.0;

    for (int c = 0; c < 94; c++) {
        double acc = (class_total[c] > 0)
                   ? 100.0 * class_correct[c] / class_total[c] : -1.0;

        std::cout << std::left
                  << std::setw(6)  << c
                  << std::setw(10) << CLASS_LABELS[c]
                  << std::setw(12) << FOLDERS[c].folder_name
                  << std::setw(9)  << class_correct[c]
                  << std::setw(8)  << class_total[c];

        if (acc < 0.0) {
            std::cout << "N/A\n";
        } else {
            std::cout << std::fixed << std::setprecision(2) << acc << "%\n";
            sum_class_acc += acc;
            classes_tested++;
            if (acc < worst_acc) { worst_acc = acc; worst_class = c; }
            if (acc > best_acc)  { best_acc  = acc; best_class  = c; }
        }
    }

    double overall_acc    = (global_total > 0)
                           ? 100.0 * global_correct / global_total : 0.0;
    double mean_class_acc = (classes_tested > 0)
                           ? (sum_class_acc / classes_tested) : 0.0;

    std::cout << "\n============================================================\n"
              << "  FINAL RESULTS\n"
              << "============================================================\n"
              << "  Total samples evaluated : " << global_total   << "\n"
              << "  Correct predictions     : " << global_correct << "\n"
              << "  Wrong  predictions      : " << (global_total - global_correct) << "\n"
              << "  Skipped (parse errors)  : " << skipped        << "\n\n"
              << "  Overall accuracy  (correct/total)  : "
              << std::fixed << std::setprecision(4) << overall_acc    << "%\n"
              << "  Mean-of-class accuracy             : "
              << std::fixed << std::setprecision(4) << mean_class_acc << "%\n\n";

    if (best_class  >= 0)
        std::cout << "  Best  class : [" << best_class  << "] '"
                  << CLASS_LABELS[best_class]  << "'  -- "
                  << std::fixed << std::setprecision(2) << best_acc  << "%\n";
    if (worst_class >= 0)
        std::cout << "  Worst class : [" << worst_class << "] '"
                  << CLASS_LABELS[worst_class] << "'  -- "
                  << std::fixed << std::setprecision(2) << worst_acc << "%\n";

    std::cout << "\n  Total inference time : " << total_ms << " ms\n"
              << "  Avg time per sample  : "
              << (global_total > 0 ? total_ms / global_total : 0.0) << " ms\n"
              << "============================================================\n"
              << "  EVALUATION COMPLETE\n"
              << "============================================================\n";

    return 0;
}