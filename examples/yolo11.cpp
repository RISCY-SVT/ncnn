// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause
// Harness-Version: 0004
// Harness-Timestamp: 2025-12-29_15-47-28

// Ultralytics YOLO11 NCNN export (format=ncnn) example.
// out0 is 2D with rows [cx, cy, w, h, class0..] over all grid cells for strides 8/16/32.

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <chrono>
#include <algorithm>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static const char kHarnessVersion[] = "0004";
static const char kHarnessTimestamp[] = "2025-12-29_15-47-28";

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct Yolo11Context
{
    ncnn::Net net;
    std::vector<int> strides;
    int target_size;
    float prob_threshold;
    float nms_threshold;
};

struct Yolo11Preproc
{
    ncnn::Mat in_pad;
    int img_w = 0;
    int img_h = 0;
    int wpad = 0;
    int hpad = 0;
    float scale = 1.f;
};

struct Yolo11Options
{
    const char* imagepath = 0;
    std::string model_dir = ".";
    std::string model_name = "yolo11n";
    std::string param_path;
    std::string bin_path;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
    int threads = 4;
    int lightmode = 1;
    int winograd = 1;
    int sgemm = 1;
    int packing = 1;
    int fp16_packed = 1;
    int fp16_storage = 1;
    int fp16_arith = 1;
    std::string pin = "cluster0";
    bool bench_only = false;
    bool no_gui = true;
    bool forward_only = false;
    int strict_omp_env = 1;
    int print_affinity = 1;
    bool print_affinity_set = false;
    bool quiet = false;
    std::string desired_omp_wait_policy;
    int desired_gomp_spincount = -1;
};

static void print_usage(const char* prog)
{
    fprintf(stderr, "Usage: %s [imagepath] [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --image <path>\n");
    fprintf(stderr, "  --model-dir <dir>\n");
    fprintf(stderr, "  --model-name <name>\n");
    fprintf(stderr, "  --param <path>\n");
    fprintf(stderr, "  --bin <path>\n");
    fprintf(stderr, "  --no-gui\n");
    fprintf(stderr, "  --gui\n");
    fprintf(stderr, "  --forward-only\n");
    fprintf(stderr, "  --bench-only\n");
    fprintf(stderr, "  --pin cluster0|none|list:<cpu_list>\n");
    fprintf(stderr, "  --warmup N\n");
    fprintf(stderr, "  --runs N\n");
    fprintf(stderr, "  --repeats N\n");
    fprintf(stderr, "  --threads N\n");
    fprintf(stderr, "  --strict-omp-env 0/1\n");
    fprintf(stderr, "  --print-affinity 0/1\n");
    fprintf(stderr, "  --quiet\n");
    fprintf(stderr, "  --desired-omp-wait-policy <PASSIVE|ACTIVE> (requires --strict-omp-env 0)\n");
    fprintf(stderr, "  --desired-gomp-spincount N (requires --strict-omp-env 0)\n");
    fprintf(stderr, "  --lightmode 0/1\n");
    fprintf(stderr, "  --winograd 0/1\n");
    fprintf(stderr, "  --sgemm 0/1\n");
    fprintf(stderr, "  --packing 0/1\n");
    fprintf(stderr, "  --fp16-packed 0/1\n");
    fprintf(stderr, "  --fp16-storage 0/1\n");
    fprintf(stderr, "  --fp16-arith 0/1\n");
}

static int parse_int_arg(const char* s, int& out)
{
    if (!s)
        return -1;
    char* end = 0;
    long v = strtol(s, &end, 10);
    if (end == s || (end && *end != '\0'))
        return -1;
    out = (int)v;
    return 0;
}

static int parse_bool_arg(const char* s, int& out)
{
    int v = 0;
    if (parse_int_arg(s, v) != 0)
        return -1;
    if (v != 0 && v != 1)
        return -1;
    out = v;
    return 0;
}

static std::string trim_copy(const std::string& s)
{
    size_t start = 0;
    while (start < s.size() && isspace(static_cast<unsigned char>(s[start])))
        start++;
    size_t end = s.size();
    while (end > start && isspace(static_cast<unsigned char>(s[end - 1])))
        end--;
    return s.substr(start, end - start);
}

static std::string join_path(const std::string& dir, const std::string& leaf)
{
    if (dir.empty() || dir == ".")
        return leaf;
    if (dir[dir.size() - 1] == '/')
        return dir + leaf;
    return dir + "/" + leaf;
}

// Resolve model file paths with precedence:
// 1) explicit --param/--bin override everything; 2) --model-dir/--model-name default to ./<name>.ncnn.*
static bool resolve_model_paths(Yolo11Options& opt, std::string& err)
{
    if (!opt.param_path.empty() || !opt.bin_path.empty())
    {
        if (opt.param_path.empty() || opt.bin_path.empty())
        {
            err = "both --param and --bin must be provided together";
            return false;
        }
        return true;
    }

    std::string base = opt.model_name + ".ncnn";
    opt.param_path = join_path(opt.model_dir, base + ".param");
    opt.bin_path = join_path(opt.model_dir, base + ".bin");
    return true;
}

static bool read_first_line(const char* path, std::string& out, std::string& err)
{
    FILE* f = fopen(path, "r");
    if (!f)
    {
        err = std::string("fopen failed: ") + path + " (" + strerror(errno) + ")";
        return false;
    }
    char buf[256];
    if (!fgets(buf, sizeof(buf), f))
    {
        fclose(f);
        err = std::string("fgets failed: ") + path;
        return false;
    }
    fclose(f);
    out = trim_copy(buf);
    return true;
}

// Detect cluster0 by asking cpu0 for its L2 shared_cpu_list (avoids hardcoding core IDs).
static bool read_cpu0_l2_shared_list(std::string& out, std::string& err)
{
    const char* base = "/sys/devices/system/cpu/cpu0/cache";
    DIR* dir = opendir(base);
    if (!dir)
    {
        err = std::string("opendir failed: ") + base + " (" + strerror(errno) + ")";
        return false;
    }
    bool found = false;
    struct dirent* ent = 0;
    while ((ent = readdir(dir)) != 0)
    {
        if (strncmp(ent->d_name, "index", 5) != 0)
            continue;
        std::string level_path = std::string(base) + "/" + ent->d_name + "/level";
        std::string level;
        std::string read_err;
        if (!read_first_line(level_path.c_str(), level, read_err))
            continue;
        if (level == "2")
        {
            std::string shared_path = std::string(base) + "/" + ent->d_name + "/shared_cpu_list";
            if (!read_first_line(shared_path.c_str(), out, read_err))
            {
                closedir(dir);
                err = read_err;
                return false;
            }
            found = true;
            break;
        }
    }
    closedir(dir);
    if (!found)
    {
        err = "L2 cache index not found for cpu0";
        return false;
    }
    return true;
}

static bool parse_int_token(const std::string& s, int& out)
{
    if (s.empty())
        return false;
    char* end = 0;
    long v = strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || (end && *end != '\0'))
        return false;
    out = static_cast<int>(v);
    return true;
}

static bool parse_cpu_list_string(const std::string& list, std::vector<int>& cpus, std::string& err)
{
    cpus.clear();
    size_t i = 0;
    while (i < list.size())
    {
        while (i < list.size() && (list[i] == ',' || isspace(static_cast<unsigned char>(list[i]))))
            i++;
        if (i >= list.size())
            break;
        size_t j = i;
        while (j < list.size() && list[j] != ',')
            j++;
        std::string token = trim_copy(list.substr(i, j - i));
        if (token.empty())
        {
            err = "empty cpu token";
            return false;
        }
        size_t dash = token.find('-');
        if (dash != std::string::npos)
        {
            std::string a_str = trim_copy(token.substr(0, dash));
            std::string b_str = trim_copy(token.substr(dash + 1));
            int a = 0;
            int b = 0;
            if (!parse_int_token(a_str, a) || !parse_int_token(b_str, b))
            {
                err = "invalid cpu range: " + token;
                return false;
            }
            if (a < 0 || b < 0 || b < a)
            {
                err = "invalid cpu range: " + token;
                return false;
            }
            for (int c = a; c <= b; c++)
                cpus.push_back(c);
        }
        else
        {
            int c = 0;
            if (!parse_int_token(token, c) || c < 0)
            {
                err = "invalid cpu id: " + token;
                return false;
            }
            cpus.push_back(c);
        }
        i = j + 1;
    }
    if (cpus.empty())
    {
        err = "cpu list is empty";
        return false;
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return true;
}

static std::string format_cpu_list(const std::vector<int>& cpus)
{
    if (cpus.empty())
        return "(none)";
    std::string out;
    size_t i = 0;
    while (i < cpus.size())
    {
        int start = cpus[i];
        int end = start;
        size_t j = i + 1;
        while (j < cpus.size() && cpus[j] == end + 1)
        {
            end = cpus[j];
            j++;
        }
        if (!out.empty())
            out += ",";
        if (start == end)
            out += std::to_string(start);
        else
            out += std::to_string(start) + "-" + std::to_string(end);
        i = j;
    }
    return out;
}

static std::string format_cpu_set(const cpu_set_t& set)
{
    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE; i++)
    {
        if (CPU_ISSET(i, &set))
            cpus.push_back(i);
    }
    return format_cpu_list(cpus);
}

static std::string read_cpus_allowed_list(pid_t tid)
{
    char path[128];
    snprintf(path, sizeof(path), "/proc/self/task/%d/status", tid);
    FILE* f = fopen(path, "r");
    if (!f)
        return "N/A";
    char line[512];
    std::string out = "N/A";
    while (fgets(line, sizeof(line), f))
    {
        if (strncmp(line, "Cpus_allowed_list:", 18) == 0)
        {
            const char* p = line + 18;
            while (*p && isspace(static_cast<unsigned char>(*p)))
                p++;
            out = trim_copy(p);
            break;
        }
    }
    fclose(f);
    return out;
}

// Apply process-wide affinity so the main thread and any non-OMP work stay on the target CPUs.
static int set_process_affinity(const std::vector<int>& cpus)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    for (size_t i = 0; i < cpus.size(); i++)
    {
        if (cpus[i] < 0 || cpus[i] >= CPU_SETSIZE)
        {
            fprintf(stderr, "cpu id %d out of range (CPU_SETSIZE=%d)\n", cpus[i], CPU_SETSIZE);
            return -1;
        }
        CPU_SET(cpus[i], &set);
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
    {
        fprintf(stderr, "sched_setaffinity failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

// Pin OpenMP worker threads in a deterministic order (thread i -> cpus[i % cpus.size()]).
static int pin_openmp_threads(const std::vector<int>& cpus, int threads, int print_affinity)
{
    if (cpus.empty())
        return -1;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    int rc = 0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Map each OpenMP thread to a single CPU in the provided list.
        int cpu = cpus[tid % cpus.size()];
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        int err = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if (err != 0)
        {
            #pragma omp critical
            {
                fprintf(stderr, "pthread_setaffinity_np failed for omp_tid=%d cpu=%d: %s\n", tid, cpu, strerror(err));
            }
            #pragma omp atomic write
            rc = -1;
        }
        #pragma omp barrier
        if (print_affinity)
        {
            cpu_set_t getset;
            CPU_ZERO(&getset);
            int geterr = pthread_getaffinity_np(pthread_self(), sizeof(getset), &getset);
            int sched_cpu = sched_getcpu();
            pid_t sys_tid = static_cast<pid_t>(syscall(SYS_gettid));
            std::string allowed = (geterr == 0) ? format_cpu_set(getset) : "N/A";
            std::string allowed_list = read_cpus_allowed_list(sys_tid);
            #pragma omp critical
            {
                fprintf(stderr,
                        "OMP thread %d (tid %d): target_cpu=%d sched_cpu=%d affinity=%s status_cpus_allowed_list=%s\n",
                        tid, sys_tid, cpu, sched_cpu, allowed.c_str(), allowed_list.c_str());
            }
        }
    }
    return rc;
#else
    (void)threads;
    if (print_affinity)
        fprintf(stderr, "OpenMP is not enabled; cannot pin OpenMP worker threads.\n");
    return 0;
#endif
}

// Enumerate OMP/GOMP environment variables so strict mode can detect external variability.
static std::vector<std::string> collect_omp_env_vars()
{
    std::vector<std::string> vars;
    extern char** environ;
    if (!environ)
        return vars;
    for (char** env = environ; *env; ++env)
    {
        const char* e = *env;
        if (strncmp(e, "OMP_", 4) == 0 || strncmp(e, "GOMP_", 5) == 0)
            vars.push_back(e);
    }
    return vars;
}

template <typename T>
static auto extractor_set_num_threads(T& ex, int num_threads, int) -> decltype(ex.set_num_threads(num_threads), void())
{
    ex.set_num_threads(num_threads);
}

template <typename T>
static void extractor_set_num_threads(T&, int, ...)
{
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_proposals_stride(const ncnn::Mat& pred, int col_offset, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int padded_w = in_pad.w;
    const int padded_h = in_pad.h;
    const int num_grid_x = padded_w / stride;
    const int num_grid_y = padded_h / stride;
    const int num_class = pred.h - 4;

    const float* cx_ptr = pred.row(0);
    const float* cy_ptr = pred.row(1);
    const float* w_ptr = pred.row(2);
    const float* h_ptr = pred.row(3);

    for (int gy = 0; gy < num_grid_y; gy++)
    {
        for (int gx = 0; gx < num_grid_x; gx++)
        {
            const int col = col_offset + gy * num_grid_x + gx;

            const float cx = cx_ptr[col];
            const float cy = cy_ptr[col];
            const float bw = w_ptr[col];
            const float bh = h_ptr[col];

            int best_label = -1;
            float best_score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                const float* score_row = pred.row(4 + k);
                const float score = score_row[col];
                if (score > best_score)
                {
                    best_score = score;
                    best_label = k;
                }
            }

            if (best_score < prob_threshold)
                continue;

            Object obj;
            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width = bw;
            obj.rect.height = bh;
            obj.label = best_label;
            obj.prob = best_score;

            objects.push_back(obj);
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    int col_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = in_pad.w / stride;
        const int num_grid_y = in_pad.h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals_stride(pred, col_offset, stride, in_pad, prob_threshold, objects);
        col_offset += num_grid;
    }
}

static int init_yolo11(Yolo11Context& ctx, const Yolo11Options& opt)
{
    ctx.strides = {8, 16, 32};
    ctx.target_size = 640;
    ctx.prob_threshold = 0.25f;
    ctx.nms_threshold = 0.45f;

    ctx.net.clear();

    ctx.net.opt.use_vulkan_compute = false;
    // Keep packing layout enabled so RVV-optimized kernels stay active.
    ctx.net.opt.use_packing_layout = opt.packing != 0;
    ctx.net.opt.use_fp16_packed = opt.fp16_packed != 0;
    ctx.net.opt.use_fp16_storage = opt.fp16_storage != 0;
    ctx.net.opt.use_fp16_arithmetic = opt.fp16_arith != 0;
    ctx.net.opt.use_bf16_storage = false;
    ctx.net.opt.use_winograd_convolution = opt.winograd != 0;
    ctx.net.opt.use_sgemm_convolution = opt.sgemm != 0;
    ctx.net.opt.lightmode = opt.lightmode != 0;
    // ctx.net.opt.use_bf16_storage = true;
    ctx.net.opt.num_threads = opt.threads;

    // https://github.com/nihui/ncnn-android-yolo11/tree/master/app/src/main/assets
    if (ctx.net.load_param(opt.param_path.c_str()) != 0)
    {
        fprintf(stderr, "load_param failed: %s\n", opt.param_path.c_str());
        return -1;
    }
    if (ctx.net.load_model(opt.bin_path.c_str()) != 0)
    {
        fprintf(stderr, "load_model failed: %s\n", opt.bin_path.c_str());
        return -1;
    }
    // yolo11.load_param("yolo11s.ncnn.param");
    // yolo11.load_model("yolo11s.ncnn.bin");
    // yolo11.load_param("yolo11m.ncnn.param");
    // yolo11.load_model("yolo11m.ncnn.bin");

    return 0;
}

static int prepare_yolo11_input(const cv::Mat& bgr, const Yolo11Context& ctx, Yolo11Preproc& prep)
{
    prep.img_w = bgr.cols;
    prep.img_h = bgr.rows;

    const int target_size = ctx.target_size;

    // letterbox pad to target_size x target_size
    int w = prep.img_w;
    int h = prep.img_h;
    prep.scale = 1.f;
    if (w > h)
    {
        prep.scale = (float)target_size / w;
        w = target_size;
        h = h * prep.scale;
    }
    else
    {
        prep.scale = (float)target_size / h;
        h = target_size;
        w = w * prep.scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, prep.img_w, prep.img_h, w, h);

    // letterbox pad to target_size rectangle
    prep.wpad = target_size - w;
    prep.hpad = target_size - h;
    if (prep.wpad < 0)
        prep.wpad = 0;
    if (prep.hpad < 0)
        prep.hpad = 0;
    ncnn::copy_make_border(in, prep.in_pad, prep.hpad / 2, prep.hpad - prep.hpad / 2, prep.wpad / 2, prep.wpad - prep.wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    prep.in_pad.substract_mean_normalize(0, norm_vals);

    return 0;
}

static int forward_yolo11(const Yolo11Context& ctx, const ncnn::Mat& in_pad, ncnn::Mat& out, int num_threads, int lightmode)
{
    ncnn::Extractor ex = ctx.net.create_extractor();
    extractor_set_num_threads(ex, num_threads, 0);
    if (lightmode)
        ex.set_light_mode(true);

    int ret_in = ex.input("in0", in_pad);
    if (ret_in != 0)
    {
        fprintf(stderr, "input in0 failed, code=%d\n", ret_in);
        return ret_in;
    }

    int ret = ex.extract("out0", out);
    if (ret != 0)
    {
        fprintf(stderr, "extract out0 failed, code=%d\n", ret);
        return ret;
    }

    return 0;
}

static void decode_yolo11(const ncnn::Mat& out, const Yolo11Preproc& prep, const Yolo11Context& ctx, std::vector<Object>& objects)
{
    std::vector<Object> proposals;
    generate_proposals(out, ctx.strides, prep.in_pad, ctx.prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, ctx.nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (prep.wpad / 2)) / prep.scale;
        float y0 = (objects[i].rect.y - (prep.hpad / 2)) / prep.scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (prep.wpad / 2)) / prep.scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (prep.hpad / 2)) / prep.scale;

        // clip
        x0 = std::max(std::min(x0, (float)(prep.img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(prep.img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(prep.img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(prep.img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

static int benchmark_yolo11_forward_only(const Yolo11Context& ctx, const ncnn::Mat& in_pad,
                                         int num_threads, int lightmode,
                                         int warmup_runs, int bench_runs, int repeats,
                                         bool print_out_shape, bool quiet)
{
    // Quiet mode suppresses per-iteration markers and shape prints while keeping summaries.
    if (bench_runs < 1 || repeats < 1)
    {
        fprintf(stderr, "bench runs and repeats must be >= 1\n");
        return -1;
    }

    std::vector<double> repeat_avgs;
    repeat_avgs.reserve(repeats);

    for (int r = 0; r < repeats; r++)
    {
        ncnn::Mat out;
        for (int i = 0; i < warmup_runs; i++)
        {
            if (!quiet)
                fprintf(stderr, "== WARMUP %d/%d ==\n", i + 1, warmup_runs);
            int ret = forward_yolo11(ctx, in_pad, out, num_threads, lightmode);
            if (ret != 0)
                return ret;
        }

        double sum_us = 0.0;
        for (int i = 0; i < bench_runs; i++)
        {
            if (!quiet)
                fprintf(stderr, "== BENCH %d/%d ==\n", i + 1, bench_runs);
            auto t0 = std::chrono::high_resolution_clock::now();
            int ret = forward_yolo11(ctx, in_pad, out, num_threads, lightmode);
            if (ret != 0)
                return ret;
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            sum_us += us;
            if (print_out_shape && !quiet && r == 0 && i == 0)
            {
                fprintf(stderr, "out0 shape: w=%d h=%d c=%d\n", out.w, out.h, out.c);
            }
        }

        double avg_us = sum_us / bench_runs;
        double fps = avg_us > 0.0 ? 1e6 / avg_us : 0.0;
        repeat_avgs.push_back(avg_us);
        fprintf(stderr, "YOLO11 pure forward benchmark: repeat %d/%d, %d runs (warmup %d)\n",
                r + 1, repeats, bench_runs, warmup_runs);
        fprintf(stderr, "Average forward time: %.2f us (%.2f FPS)\n", avg_us, fps);
    }

    double mean = 0.0;
    for (size_t i = 0; i < repeat_avgs.size(); i++)
        mean += repeat_avgs[i];
    mean /= repeat_avgs.size();
    double var = 0.0;
    for (size_t i = 0; i < repeat_avgs.size(); i++)
    {
        double diff = repeat_avgs[i] - mean;
        var += diff * diff;
    }
    var /= repeat_avgs.size();
    double stddev = sqrt(var);
    fprintf(stderr, "Repeat summary: mean %.2f us, stddev %.2f us (n=%d)\n",
            mean, stddev, repeats);

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static cv::Scalar colors[] = {
        cv::Scalar(244, 67, 54),
        cv::Scalar(233, 30, 99),
        cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181),
        cv::Scalar(33, 150, 243),
        cv::Scalar(3, 169, 244),
        cv::Scalar(0, 188, 212),
        cv::Scalar(0, 150, 136),
        cv::Scalar(76, 175, 80),
        cv::Scalar(139, 195, 74),
        cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59),
        cv::Scalar(255, 193, 7),
        cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34),
        cv::Scalar(121, 85, 72),
        cv::Scalar(158, 158, 158),
        cv::Scalar(96, 125, 139)
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    Yolo11Options opt;

    // Parse CLI options; explicit flags override defaults and may affect later validation logic.
    for (int i = 1; i < argc; i++)
    {
        const char* arg = argv[i];
        if (strcmp(arg, "--image") == 0)
        {
            if (i + 1 >= argc)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.imagepath = argv[++i];
            continue;
        }
        if (strcmp(arg, "--model-dir") == 0)
        {
            if (i + 1 >= argc)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.model_dir = argv[++i];
            continue;
        }
        if (strcmp(arg, "--model-name") == 0)
        {
            if (i + 1 >= argc)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.model_name = argv[++i];
            continue;
        }
        if (strcmp(arg, "--param") == 0)
        {
            if (i + 1 >= argc)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.param_path = argv[++i];
            continue;
        }
        if (strcmp(arg, "--bin") == 0)
        {
            if (i + 1 >= argc)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.bin_path = argv[++i];
            continue;
        }
        if (strcmp(arg, "--no-gui") == 0)
        {
            opt.no_gui = true;
            continue;
        }
        if (strcmp(arg, "--gui") == 0)
        {
            opt.no_gui = false;
            continue;
        }
        if (strcmp(arg, "--forward-only") == 0)
        {
            opt.forward_only = true;
            continue;
        }
        if (strcmp(arg, "--bench-only") == 0)
        {
            opt.bench_only = true;
            opt.forward_only = true;
            opt.no_gui = true;
            continue;
        }
        if (strcmp(arg, "--pin") == 0)
        {
            if (i + 1 >= argc)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.pin = argv[++i];
            continue;
        }
        if (strcmp(arg, "--warmup") == 0)
        {
            if (i + 1 >= argc || parse_int_arg(argv[i + 1], opt.warmup) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--runs") == 0)
        {
            if (i + 1 >= argc || parse_int_arg(argv[i + 1], opt.runs) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--repeats") == 0)
        {
            if (i + 1 >= argc || parse_int_arg(argv[i + 1], opt.repeats) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--threads") == 0)
        {
            if (i + 1 >= argc || parse_int_arg(argv[i + 1], opt.threads) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--strict-omp-env") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.strict_omp_env) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--print-affinity") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.print_affinity) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.print_affinity_set = true;
            i++;
            continue;
        }
        if (strcmp(arg, "--quiet") == 0)
        {
            // Quiet mode is for performance runs: keep summaries, drop per-iteration noise.
            opt.quiet = true;
            continue;
        }
        if (strcmp(arg, "--desired-omp-wait-policy") == 0)
        {
            if (i + 1 >= argc)
            {
                print_usage(argv[0]);
                return -1;
            }
            opt.desired_omp_wait_policy = argv[++i];
            continue;
        }
        if (strcmp(arg, "--desired-gomp-spincount") == 0)
        {
            if (i + 1 >= argc || parse_int_arg(argv[i + 1], opt.desired_gomp_spincount) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--lightmode") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.lightmode) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--winograd") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.winograd) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--sgemm") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.sgemm) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--packing") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.packing) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--fp16-packed") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.fp16_packed) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--fp16-storage") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.fp16_storage) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--fp16-arith") == 0)
        {
            if (i + 1 >= argc || parse_bool_arg(argv[i + 1], opt.fp16_arith) != 0)
            {
                print_usage(argv[0]);
                return -1;
            }
            i++;
            continue;
        }

        if (arg[0] != '-')
        {
            opt.imagepath = arg;
            continue;
        }

        fprintf(stderr, "Unknown option: %s\n", arg);
        print_usage(argv[0]);
        return -1;
    }

    if (!opt.imagepath)
    {
        print_usage(argv[0]);
        return -1;
    }
    if (opt.warmup < 0 || opt.runs < 0)
    {
        fprintf(stderr, "warmup and runs must be >= 0\n");
        return -1;
    }
    if (opt.repeats < 1)
    {
        fprintf(stderr, "repeats must be >= 1\n");
        return -1;
    }
    if (opt.threads < 1)
    {
        fprintf(stderr, "threads must be >= 1\n");
        return -1;
    }
    if (opt.bench_only && opt.runs < 1)
    {
        fprintf(stderr, "bench-only requires --runs >= 1\n");
        return -1;
    }

    // Quiet mode disables affinity printing by default unless the user asked for it explicitly.
    if (opt.quiet && !opt.print_affinity_set)
        opt.print_affinity = 0;

    const bool desired_opts_set = !opt.desired_omp_wait_policy.empty() || opt.desired_gomp_spincount >= 0;
    if (opt.strict_omp_env && desired_opts_set)
    {
        // Strict mode forbids desired-* hints because they rely on external OMP/GOMP env vars.
        fprintf(stderr, "ERROR: --strict-omp-env=1 forbids --desired-* options (they imply external env vars)\n");
        fprintf(stderr, "Disable strict mode (--strict-omp-env 0) or remove desired-* options.\n");
        return -1;
    }

    // Resolve model files after parsing so --param/--bin can override model-dir/name.
    std::string model_err;
    if (!resolve_model_paths(opt, model_err))
    {
        fprintf(stderr, "ERROR: %s\n", model_err.c_str());
        return -1;
    }

    // Environment hygiene: fail fast if external OMP/GOMP vars are set in strict mode.
    std::vector<std::string> omp_env = collect_omp_env_vars();
    fprintf(stderr, "ENVIRONMENT (OMP/GOMP):\n");
    if (omp_env.empty())
    {
        fprintf(stderr, "  (none)\n");
    }
    else
    {
        for (size_t i = 0; i < omp_env.size(); i++)
            fprintf(stderr, "  %s\n", omp_env[i].c_str());
    }
    if (opt.strict_omp_env && !omp_env.empty())
    {
        fprintf(stderr, "ERROR: OMP_/GOMP_ environment variables are set while --strict-omp-env=1\n");
        fprintf(stderr, "Unset them or pass --strict-omp-env=0 to continue.\n");
        return -1;
    }

    // Resolve cluster0 list from sysfs
    std::string cluster0_list;
    std::string cluster_err;
    if (!read_cpu0_l2_shared_list(cluster0_list, cluster_err))
    {
        fprintf(stderr, "ERROR: failed to read cluster0 list: %s\n", cluster_err.c_str());
        return -1;
    }
    std::vector<int> cluster0_cpus;
    if (!parse_cpu_list_string(cluster0_list, cluster0_cpus, cluster_err))
    {
        fprintf(stderr, "ERROR: failed to parse cluster0 list '%s': %s\n", cluster0_list.c_str(), cluster_err.c_str());
        return -1;
    }

    std::vector<int> pin_cpus;
    if (opt.pin == "none")
    {
        fprintf(stderr, "WARNING: --pin none disables CPU affinity and may reduce determinism.\n");
    }
    else
    {
        std::string pin_list;
        if (opt.pin == "cluster0")
        {
            pin_cpus = cluster0_cpus;
        }
        else
        {
            if (opt.pin.rfind("list:", 0) == 0)
                pin_list = opt.pin.substr(5);
            else
                pin_list = opt.pin;
            if (!parse_cpu_list_string(pin_list, pin_cpus, cluster_err))
            {
                fprintf(stderr, "ERROR: failed to parse --pin list '%s': %s\n", pin_list.c_str(), cluster_err.c_str());
                return -1;
            }
            for (size_t i = 0; i < pin_cpus.size(); i++)
            {
                if (std::find(cluster0_cpus.begin(), cluster0_cpus.end(), pin_cpus[i]) == cluster0_cpus.end())
                {
                    fprintf(stderr, "ERROR: --pin list includes CPU %d outside cluster0 (%s)\n",
                            pin_cpus[i], format_cpu_list(cluster0_cpus).c_str());
                    return -1;
                }
            }
        }
    }
    if (!pin_cpus.empty() && opt.threads > static_cast<int>(pin_cpus.size()))
    {
        fprintf(stderr, "WARNING: threads (%d) exceed pinned CPUs (%zu); threads will share CPUs.\n",
                opt.threads, pin_cpus.size());
    }

    std::string param_lower = opt.param_path;
    std::transform(param_lower.begin(), param_lower.end(), param_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    const int model_int8_hint = param_lower.find("int8") != std::string::npos ? 1 : 0;
    ncnn::Option int8_opt;

    fprintf(stderr, "Effective options:\n");
    fprintf(stderr, "  harness_version: %s\n", kHarnessVersion);
    fprintf(stderr, "  harness_timestamp: %s\n", kHarnessTimestamp);
    fprintf(stderr, "  image: %s\n", opt.imagepath);
    fprintf(stderr, "  model_dir: %s\n", opt.model_dir.c_str());
    fprintf(stderr, "  model_name: %s\n", opt.model_name.c_str());
    fprintf(stderr, "  param_path: %s\n", opt.param_path.c_str());
    fprintf(stderr, "  bin_path: %s\n", opt.bin_path.c_str());
    fprintf(stderr, "  pin: %s\n", opt.pin.c_str());
    fprintf(stderr, "  cluster0: %s\n", format_cpu_list(cluster0_cpus).c_str());
    fprintf(stderr, "  pin_cpus: %s\n", pin_cpus.empty() ? "(none)" : format_cpu_list(pin_cpus).c_str());
    fprintf(stderr, "  no_gui: %d\n", opt.no_gui ? 1 : 0);
    fprintf(stderr, "  forward_only: %d\n", opt.forward_only ? 1 : 0);
    fprintf(stderr, "  bench_only: %d\n", opt.bench_only ? 1 : 0);
    fprintf(stderr, "  warmup: %d\n", opt.warmup);
    fprintf(stderr, "  runs: %d\n", opt.runs);
    fprintf(stderr, "  repeats: %d\n", opt.repeats);
    fprintf(stderr, "  threads: %d\n", opt.threads);
    fprintf(stderr, "  strict_omp_env: %d\n", opt.strict_omp_env);
    fprintf(stderr, "  print_affinity: %d\n", opt.print_affinity);
    fprintf(stderr, "  quiet: %d\n", opt.quiet ? 1 : 0);
    fprintf(stderr, "  desired_omp_wait_policy: %s\n",
            opt.desired_omp_wait_policy.empty() ? "(unspecified)" : opt.desired_omp_wait_policy.c_str());
    fprintf(stderr, "  desired_gomp_spincount: %d\n", opt.desired_gomp_spincount);
    fprintf(stderr, "  lightmode: %d\n", opt.lightmode);
    fprintf(stderr, "  winograd: %d\n", opt.winograd);
    fprintf(stderr, "  sgemm: %d\n", opt.sgemm);
    fprintf(stderr, "  packing: %d\n", opt.packing);
    fprintf(stderr, "  fp16_packed: %d\n", opt.fp16_packed);
    fprintf(stderr, "  fp16_storage: %d\n", opt.fp16_storage);
    fprintf(stderr, "  fp16_arith: %d\n", opt.fp16_arith);
#ifdef NCNN_INT8
    fprintf(stderr, "  int8_build_support: %d\n", NCNN_INT8 ? 1 : 0);
#else
    fprintf(stderr, "  int8_build_support: unknown (no macro)\n");
#endif
    fprintf(stderr, "  int8_inference: %d\n", int8_opt.use_int8_inference);
    fprintf(stderr, "  int8_packed: %d\n", int8_opt.use_int8_packed);
    fprintf(stderr, "  int8_storage: %d\n", int8_opt.use_int8_storage);
    fprintf(stderr, "  int8_arithmetic: %d\n", int8_opt.use_int8_arithmetic);
    fprintf(stderr, "  int8_uniform: %d\n", int8_opt.use_int8_uniform);
    fprintf(stderr, "  model_int8_hint: %d\n", model_int8_hint);
    fprintf(stderr, "\n");

    if (!opt.desired_omp_wait_policy.empty() || opt.desired_gomp_spincount >= 0)
    {
        fprintf(stderr, "NOTE: OpenMP wait policy is not set in-code. To apply it, set env vars before launch:\n");
        fprintf(stderr, "  OMP_WAIT_POLICY=%s GOMP_SPINCOUNT=%d %s [args...]\n",
                opt.desired_omp_wait_policy.empty() ? "PASSIVE" : opt.desired_omp_wait_policy.c_str(),
                opt.desired_gomp_spincount >= 0 ? opt.desired_gomp_spincount : 0,
                argv[0]);
    }

    // Apply process and thread affinity only when pinning is enabled.
    if (!pin_cpus.empty())
    {
        if (set_process_affinity(pin_cpus) != 0)
            return -1;
        if (opt.print_affinity)
        {
            cpu_set_t set;
            CPU_ZERO(&set);
            if (sched_getaffinity(0, sizeof(set), &set) == 0)
                fprintf(stderr, "Process affinity (after set): %s\n", format_cpu_set(set).c_str());
            else
                fprintf(stderr, "sched_getaffinity failed: %s\n", strerror(errno));
        }
        if (pin_openmp_threads(pin_cpus, opt.threads, opt.print_affinity) != 0)
            return -1;
        fprintf(stderr, "\n");
    }

    const char* imagepath = opt.imagepath;

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    Yolo11Context ctx;
    if (init_yolo11(ctx, opt) != 0)
        return -1;

    auto t_total0 = std::chrono::high_resolution_clock::now();

    Yolo11Preproc prep;
    if (prepare_yolo11_input(m, ctx, prep) != 0)
    {
        fprintf(stderr, "prepare_yolo11_input failed\n");
        return -1;
    }

    if (opt.bench_only)
    {
        // Bench-only: only warmup + timed forward passes (no decode/draw).
        if (benchmark_yolo11_forward_only(ctx, prep.in_pad, opt.threads, opt.lightmode,
                                          opt.warmup, opt.runs, opt.repeats, true, opt.quiet) != 0)
        {
            fprintf(stderr, "benchmark_yolo11_forward_only failed\n");
            return -1;
        }
        return 0;
    }

    std::vector<Object> objects;
    double pure_us = 0.0;
    double total_us = 0.0;
    auto t_fwd0 = std::chrono::high_resolution_clock::now();
    ncnn::Mat out;
    int ret = forward_yolo11(ctx, prep.in_pad, out, opt.threads, opt.lightmode);
    if (ret != 0)
    {
        fprintf(stderr, "forward_yolo11 failed, code=%d\n", ret);
        return -1;
    }
    if (!opt.quiet)
        fprintf(stderr, "out0 shape: w=%d h=%d c=%d\n", out.w, out.h, out.c);
    auto t_fwd1 = std::chrono::high_resolution_clock::now();

    auto t_total1 = t_fwd1;
    if (!opt.forward_only)
    {
        decode_yolo11(out, prep, ctx, objects);
        t_total1 = std::chrono::high_resolution_clock::now();
    }

    pure_us = std::chrono::duration<double, std::micro>(t_fwd1 - t_fwd0).count();
    total_us = std::chrono::duration<double, std::micro>(t_total1 - t_total0).count();

    double pure_fps = pure_us > 0.0 ? 1e6 / pure_us : 0.0;
    double total_fps = total_us > 0.0 ? 1e6 / total_us : 0.0;
    if (!opt.quiet)
    {
        fprintf(stderr, "Model forward time: %.2f us (%.2f FPS)\n", pure_us, pure_fps);
        fprintf(stderr, "Total frame time:  %.2f us (%.2f FPS)\n", total_us, total_fps);
    }

    if (opt.runs > 0 && benchmark_yolo11_forward_only(ctx, prep.in_pad, opt.threads, opt.lightmode,
                                                      opt.warmup, opt.runs, opt.repeats, false, opt.quiet) != 0)
    {
        fprintf(stderr, "benchmark_yolo11_forward_only failed\n");
    }

    if (!opt.forward_only && !opt.no_gui)
        draw_objects(m, objects);

    return 0;
}
