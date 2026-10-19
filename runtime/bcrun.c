/*
 * bcrun.c — moteur d'inférence C99 pour les paquets BlackCode (.bcpkg).
 *
 * Objectif : mesurer l'inférence long-contexte sur machine à 4 Go de RAM.
 * Les poids sont mappés en lecture seule ; le pic dépend surtout du cache KV
 * et du contexte. Le runtime lui-même reste léger, hors poids et cache KV.
 *
 * Caractéristiques :
 *   - mmap du fichier : les poids ne sont jamais copiés en RAM anonyme ;
 *   - GEMV ternaire (2 bits/poids) décodé à la volée ;
 *   - embeddings int8 + échelle par ligne ;
 *   - RMSNorm, RoPE, GQA avec cache KV fp16, convolutions courtes depthwise ;
 *   - génération gloutonne ou top-k.
 *
 * Compilation :
 *   cc -O3 -march=native -o bcrun bcrun.c -lm
 *
 * Usage :
 *   ./bcrun --pkg m.bcpkg --ids 1,2,3 --max-new 8 --context 4096 --mem-report
 *   ./bcrun --pkg m.bcpkg --ids 1,2 --dump-logits logits.f32
 */
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>

#define MAX_TENSORS 4096
#define MAX_LAYERS 128
#define MAX_NAME 128

/* ------------------------------------------------------------------ */
/* utilitaires                                                         */
/* ------------------------------------------------------------------ */
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double rss_mb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0; /* Linux : Ko */
}

static void die(const char *msg) {
    fprintf(stderr, "bcrun: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("malloc a échoué");
    return p;
}

/* fp16 <-> fp32 (portable, sans _Float16) */
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3ffu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (man << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static uint16_t f2h(float v) {
    uint32_t f;
    memcpy(&f, &v, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
    uint32_t man = f & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

/* ------------------------------------------------------------------ */
/* mini parseur JSON (suffisant pour nos headers)                      */
/* ------------------------------------------------------------------ */
static char *skip_ws(char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

static char *skip_string(char *p) {
    p++; /* " */
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p++;
        p++;
    }
    return (*p == '"') ? p + 1 : p;
}

static char *skip_value(char *p) {
    p = skip_ws(p);
    if (*p == '"') return skip_string(p);
    if (*p == '{' || *p == '[') {
        char open = *p, close = (open == '{') ? '}' : ']';
        int depth = 0;
        while (*p) {
            if (*p == '"') { p = skip_string(p); continue; }
            if (*p == open) depth++;
            else if (*p == close) { depth--; if (depth == 0) return p + 1; }
            p++;
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

static int json_find(char *obj, const char *key, char **out) {
    char *p = skip_ws(obj);
    if (*p != '{') return 0;
    p++;
    size_t klen = strlen(key);
    for (;;) {
        p = skip_ws(p);
        if (*p == '}') return 0;
        if (*p != '"') return 0;
        char *kstart = p + 1;
        p = skip_string(p);
        size_t len = (size_t)(p - 1 - kstart);
        char *q = skip_ws(p);
        if (*q == ':') q = skip_ws(q + 1);
        if (len == klen && strncmp(kstart, key, klen) == 0) { *out = q; return 1; }
        p = skip_value(q);
        p = skip_ws(p);
        if (*p == ',') p++;
    }
}

static long json_int(char *obj, const char *key, long dflt) {
    char *v;
    if (!json_find(obj, key, &v)) return dflt;
    return strtol(v, NULL, 10);
}

static double json_double(char *obj, const char *key, double dflt) {
    char *v;
    if (!json_find(obj, key, &v)) return dflt;
    return strtod(v, NULL);
}

static int json_str(char *obj, const char *key, char *buf, size_t n) {
    char *v;
    if (!json_find(obj, key, &v) || *v != '"') return 0;
    char *s = v + 1, *e = skip_string(v) - 1;
    size_t len = (size_t)(e - s);
    if (len >= n) len = n - 1;
    memcpy(buf, s, len);
    buf[len] = 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/* tenseurs                                                            */
/* ------------------------------------------------------------------ */
typedef enum { DT_TERNARY = 1, DT_INT8ROW = 2, DT_FP16 = 3 } DType;

typedef struct {
    char name[MAX_NAME];
    DType dtype;
    int rows, cols;          /* 2D : [rows][cols] ; 1D : rows=1 */
    int ndim;
    int shape[4];
    const uint8_t *data;     /* pointeur dans le mmap */
    const uint16_t *scales;  /* fp16, une par ligne */
    int packed_cols;
} Tensor;

typedef struct {
    int vocab, d_model, n_layers, n_heads, n_kv, head_dim, ffn, kconv;
    int attn_every, loop, max_seq, rope_original_max_seq;
    float eps, rope_base, rope_factor, rope_beta_fast, rope_beta_slow;
    char rope_scaling[16];
    Tensor tensors[MAX_TENSORS];
    int n_tensors;
    const char *map;
    size_t map_size;
} Package;

static Tensor *find_tensor(Package *p, const char *name) {
    for (int i = 0; i < p->n_tensors; i++)
        if (strcmp(p->tensors[i].name, name) == 0) return &p->tensors[i];
    fprintf(stderr, "bcrun: tenseur introuvable : %s\n", name);
    exit(1);
}

static void load_package(Package *p, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("impossible d'ouvrir le paquet");
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat a échoué");
    p->map_size = (size_t)st.st_size;
    p->map = mmap(NULL, p->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p->map == MAP_FAILED) die("mmap a échoué");
    close(fd);
    madvise((void *)p->map, p->map_size, MADV_WILLNEED);

    if (memcmp(p->map, "BCPK1\n", 6) != 0) die("magic invalide");
    uint32_t hlen;
    memcpy(&hlen, p->map + 6, 4);
    /* le mmap est en lecture seule : on copie le header (quelques centaines
       de Ko) dans un tampon mutable pour le terminer par un zéro. */
    char *header = xmalloc((size_t)hlen + 1);
    memcpy(header, p->map + 10, hlen);
    header[hlen] = 0;

    char *cfg;
    if (!json_find(header, "config", &cfg)) die("config absente");
    p->vocab = (int)json_int(cfg, "vocab_size", 32768);
    p->d_model = (int)json_int(cfg, "d_model", 2048);
    p->n_layers = (int)json_int(cfg, "n_layers", 24);
    p->n_heads = (int)json_int(cfg, "n_heads", 16);
    p->n_kv = (int)json_int(cfg, "n_kv_heads", 4);
    p->head_dim = (int)json_int(cfg, "head_dim", 128);
    p->ffn = (int)json_int(cfg, "ffn_hidden", 5632);
    p->kconv = (int)json_int(cfg, "conv_kernel", 4);
    p->attn_every = (int)json_int(cfg, "attn_every", 4);
    p->loop = (int)json_int(cfg, "loop", 1);
    p->max_seq = (int)json_int(cfg, "max_seq_len", 8192);
    p->eps = (float)json_double(cfg, "norm_eps", 1e-5);
    p->rope_base = (float)json_double(cfg, "rope_base", 10000.0);
    p->rope_scaling[0] = 0;
    json_str(cfg, "rope_scaling", p->rope_scaling, sizeof(p->rope_scaling));
    if (!p->rope_scaling[0]) strcpy(p->rope_scaling, "none");
    p->rope_original_max_seq = (int)json_int(cfg, "rope_original_max_seq_len", 16384);
    p->rope_factor = (float)json_double(cfg, "rope_factor", 1.0);
    p->rope_beta_fast = (float)json_double(cfg, "rope_beta_fast", 32.0);
    p->rope_beta_slow = (float)json_double(cfg, "rope_beta_slow", 1.0);

    char *arr;
    if (!json_find(header, "tensors", &arr) || *arr != '[') die("tensors absents");
    arr++;
    p->n_tensors = 0;
    for (;;) {
        arr = skip_ws(arr);
        if (*arr == ']') break;
        if (*arr != '{') die("entrée de tenseur invalide");
        if (p->n_tensors >= MAX_TENSORS) die("trop de tenseurs");
        Tensor *t = &p->tensors[p->n_tensors++];
        json_str(arr, "name", t->name, MAX_NAME);
        char type[16] = {0};
        json_str(arr, "dtype", type, sizeof(type));
        if (strcmp(type, "ternary2") == 0) t->dtype = DT_TERNARY;
        else if (strcmp(type, "int8row") == 0) t->dtype = DT_INT8ROW;
        else t->dtype = DT_FP16;

        char *shp;
        t->ndim = 0;
        if (json_find(arr, "shape", &shp) && *shp == '[') {
            shp++;
            while (*shp && *shp != ']') {
                if (t->ndim < 4) t->shape[t->ndim++] = (int)strtol(shp, &shp, 10);
                shp = skip_ws(shp);
                if (*shp == ',') shp++;
            }
        }
        if (t->dtype == DT_FP16) {
            t->rows = (t->ndim >= 1) ? t->shape[0] : 1;
            t->cols = (t->ndim >= 2) ? t->shape[1] : 1;
        } else {
            t->rows = (int)json_int(arr, "shape" , 0) ; /* remplacé dessous */
            t->rows = (int)json_int(arr, "in_features", 0); /* placeholder */
            /* rows = shape[0] */
            t->rows = t->shape[0];
            t->cols = (int)json_int(arr, "in_features", t->shape[1]);
            t->packed_cols = (int)json_int(arr, "packed_cols", (t->cols + 3) / 4);
        }
        long off = json_int(arr, "offset", -1);
        if (off < 0) die("offset manquant");
        t->data = (const uint8_t *)p->map + 10 + hlen + off;
        long soff = json_int(arr, "scale_offset", -1);
        t->scales = (soff >= 0)
            ? (const uint16_t *)((const uint8_t *)p->map + 10 + hlen + soff)
            : NULL;

        arr = skip_value(arr);
        arr = skip_ws(arr);
        if (*arr == ',') arr++;
    }
    free(header);
}

/* ------------------------------------------------------------------ */
/* opérations                                                          */
/* ------------------------------------------------------------------ */
static void rmsnorm(float *out, const float *x, const Tensor *w, int n, float eps) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += x[i] * x[i];
    float scale = 1.0f / sqrtf(acc / (float)n + eps);
    /* les normes sont fp16 : on lit les valeurs directement */
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * h2f(((const uint16_t *)w->data)[i]);
}

/* y[rows] = scale[r] * sum_c ternaire(r,c) * x[c] */
static void ternary_matvec(const Tensor *t, const float *x, float *y) {
    const int rows = t->rows, cols = t->cols, pc = t->packed_cols;
    const int full = cols & ~3;   /* multiple de 4 : pas de branche dans le coeur */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = t->data + (size_t)r * pc;
        /* 4 accumulateurs indépendants : casse la dépendance de chaîne et
           laisse l'auto-vectorisation faire son travail. */
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int c = 0, b = 0;
        for (; c < full; c += 4, b++) {
            uint8_t byte = row[b];
            a0 += (float)((int)(byte & 3u) - 1) * x[c];
            a1 += (float)((int)((byte >> 2) & 3u) - 1) * x[c + 1];
            a2 += (float)((int)((byte >> 4) & 3u) - 1) * x[c + 2];
            a3 += (float)((int)((byte >> 6) & 3u) - 1) * x[c + 3];
        }
        float acc = (a0 + a1) + (a2 + a3);
        for (; c < cols; c++) {
            int code = (row[c >> 2] >> (2 * (c & 3))) & 3;
            acc += (float)(code - 1) * x[c];
        }
        float sc = t->scales ? h2f(t->scales[r]) : 1.0f;
        y[r] = acc * sc;
    }
}

static void int8_matvec(const Tensor *t, const float *x, float *y) {
    const int rows = t->rows, cols = t->cols;
    const int8_t *w = (const int8_t *)t->data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const int8_t *row = w + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += (float)row[c] * x[c];
        y[r] = acc * (t->scales ? h2f(t->scales[r]) : 1.0f);
    }
}

static void fp16_matvec(const Tensor *t, const float *x, float *y) {
    const int rows = t->rows, cols = t->cols;
    const uint16_t *w = (const uint16_t *)t->data;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = w + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += h2f(row[c]) * x[c];
        y[r] = acc;
    }
}

static void matvec(const Tensor *t, const float *x, float *y) {
    switch (t->dtype) {
        case DT_TERNARY: ternary_matvec(t, x, y); break;
        case DT_INT8ROW: int8_matvec(t, x, y); break;
        default: fp16_matvec(t, x, y); break;
    }
}

static void embedding_row(const Tensor *t, int id, float *out) {
    if (t->dtype == DT_INT8ROW) {
        const int8_t *row = (const int8_t *)t->data + (size_t)id * t->cols;
        float sc = h2f(t->scales[id]);
        for (int c = 0; c < t->cols; c++) out[c] = (float)row[c] * sc;
    } else {
        const uint16_t *row = (const uint16_t *)t->data + (size_t)id * t->cols;
        for (int c = 0; c < t->cols; c++) out[c] = h2f(row[c]);
    }
}

static void silu_mul(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] = (a[i] / (1.0f + expf(-a[i]))) * b[i];
}

/* ------------------------------------------------------------------ */
/* modèle                                                              */
/* ------------------------------------------------------------------ */
typedef struct {
    int is_attn;
    Tensor *norm, *ffn_norm;
    Tensor *q, *k, *v, *o, *qnorm, *knorm;
    Tensor *in_proj, *conv, *out_proj;
    Tensor *w1, *w3, *w2;
} Block;

typedef struct {
    Package *pkg;
    Block blocks[MAX_LAYERS];
    int n_blocks_total;   /* n_layers * loop */
    Tensor *emb, *norm_f;
    int ctx;
    uint16_t *kv;         /* [2][n_layers_attn][ctx][n_kv][head_dim] */
    int n_attn;
    int *attn_layer_index;/* index de slot KV pour chaque bloc */
    float *conv_cache;    /* [n_conv][k-1][d] */
    int *conv_slot;
    int n_conv;
    float *rope_cos, *rope_sin; /* [ctx][head_dim/2] */
    float *x, *h, *a, *b, *gate, *q, *k, *v, *att, *tmp;
} Model;

static double yarn_correction_dim(double rotations, int dim, double base,
                                  int original_max_seq) {
    return (dim * log((double)original_max_seq / (rotations * 2.0 * M_PI)))
           / (2.0 * log(base));
}

static void model_init(Model *m, Package *pkg, int ctx) {
    memset(m, 0, sizeof(*m));
    m->pkg = pkg;
    m->ctx = ctx;
    m->emb = find_tensor(pkg, "tok_emb.weight");
    m->norm_f = find_tensor(pkg, "norm_f.weight");

    const int d = pkg->d_model, f = pkg->ffn;
    m->x = xmalloc(sizeof(float) * d);
    m->h = xmalloc(sizeof(float) * d);
    m->a = xmalloc(sizeof(float) * f);
    m->b = xmalloc(sizeof(float) * f);
    m->gate = xmalloc(sizeof(float) * (d > f ? d : f));
    m->q = xmalloc(sizeof(float) * pkg->n_heads * pkg->head_dim);
    m->k = xmalloc(sizeof(float) * pkg->n_kv * pkg->head_dim);
    m->v = xmalloc(sizeof(float) * pkg->n_kv * pkg->head_dim);
    m->att = xmalloc(sizeof(float) * ctx);
    m->tmp = xmalloc(sizeof(float) * (d > f ? d : f));

    m->n_blocks_total = pkg->n_layers * pkg->loop;
    m->n_attn = 0;
    m->n_conv = 0;
    m->attn_layer_index = xmalloc(sizeof(int) * m->n_blocks_total);
    m->conv_slot = xmalloc(sizeof(int) * m->n_blocks_total);

    for (int slot = 0; slot < m->n_blocks_total; slot++) {
        int layer = slot % pkg->n_layers;
        char name[MAX_NAME];
        Block *b = &m->blocks[slot];
        b->is_attn = ((layer + 1) % pkg->attn_every) == 0;
        if (b->is_attn) {
            snprintf(name, MAX_NAME, "blocks.%d.attn_norm.weight", layer);
            b->norm = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.ffn_norm.weight", layer);
            b->ffn_norm = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.attn.q_proj.weight", layer);
            b->q = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.attn.k_proj.weight", layer);
            b->k = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.attn.v_proj.weight", layer);
            b->v = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.attn.o_proj.weight", layer);
            b->o = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.attn.q_norm.weight", layer);
            b->qnorm = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.attn.k_norm.weight", layer);
            b->knorm = find_tensor(pkg, name);
            m->attn_layer_index[slot] = m->n_attn++;
        } else {
            snprintf(name, MAX_NAME, "blocks.%d.norm.weight", layer);
            b->norm = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.ffn_norm.weight", layer);
            b->ffn_norm = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.in_proj.weight", layer);
            b->in_proj = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.conv.weight", layer);
            b->conv = find_tensor(pkg, name);
            snprintf(name, MAX_NAME, "blocks.%d.out_proj.weight", layer);
            b->out_proj = find_tensor(pkg, name);
            m->conv_slot[slot] = m->n_conv++;
        }
        snprintf(name, MAX_NAME, "blocks.%d.ffn.w1.weight", layer);
        b->w1 = find_tensor(pkg, name);
        snprintf(name, MAX_NAME, "blocks.%d.ffn.w3.weight", layer);
        b->w3 = find_tensor(pkg, name);
        snprintf(name, MAX_NAME, "blocks.%d.ffn.w2.weight", layer);
        b->w2 = find_tensor(pkg, name);
    }

    size_t kv_elems = (size_t)2 * m->n_attn * ctx * pkg->n_kv * pkg->head_dim;
    m->kv = xmalloc(kv_elems * sizeof(uint16_t));
    memset(m->kv, 0, kv_elems * sizeof(uint16_t)); /* touche les pages */

    size_t conv_elems = (size_t)m->n_conv * (pkg->kconv - 1) * d;
    m->conv_cache = xmalloc(conv_elems * sizeof(float));
    memset(m->conv_cache, 0, conv_elems * sizeof(float));

    int half = pkg->head_dim / 2;
    double yarn_low = 0.0, yarn_high = 0.0, rope_attention_factor = 1.0;
    if (strcmp(pkg->rope_scaling, "none") == 0) {
        /* Standard RoPE; no scaling. */
    } else if (strcmp(pkg->rope_scaling, "yarn") == 0) {
        if (pkg->rope_original_max_seq < 1 || pkg->rope_factor <= 1.0f ||
            pkg->rope_beta_fast <= pkg->rope_beta_slow || pkg->rope_beta_slow <= 0.0f)
            die("paramètres YaRN invalides");
        yarn_low = floor(yarn_correction_dim(pkg->rope_beta_fast, pkg->head_dim,
                                             pkg->rope_base, pkg->rope_original_max_seq));
        yarn_high = ceil(yarn_correction_dim(pkg->rope_beta_slow, pkg->head_dim,
                                             pkg->rope_base, pkg->rope_original_max_seq));
        if (yarn_low < 0.0) yarn_low = 0.0;
        if (yarn_low > half - 1) yarn_low = half - 1;
        if (yarn_high < 0.0) yarn_high = 0.0;
        if (yarn_high > half - 1) yarn_high = half - 1;
        if (yarn_high == yarn_low && yarn_high < half - 1) yarn_high += 1.0;
        rope_attention_factor = 1.0 + 0.1 * log((double)pkg->rope_factor);
    } else {
        die("rope_scaling inconnu (attendu none ou yarn)");
    }

    m->rope_cos = xmalloc(sizeof(float) * (size_t)ctx * half);
    m->rope_sin = xmalloc(sizeof(float) * (size_t)ctx * half);
    for (int pos = 0; pos < ctx; pos++) {
        for (int i = 0; i < half; i++) {
            double inv = pow((double)pkg->rope_base, -2.0 * i / pkg->head_dim);
            if (strcmp(pkg->rope_scaling, "yarn") == 0) {
                double ramp = ((double)i - yarn_low) / (yarn_high - yarn_low);
                if (ramp < 0.0) ramp = 0.0;
                if (ramp > 1.0) ramp = 1.0;
                /* Smoothly mix original extrapolation and scaled interpolation. */
                inv = inv * (1.0 - ramp) + (inv / pkg->rope_factor) * ramp;
            }
            double ang = pos * inv;
            m->rope_cos[(size_t)pos * half + i] = (float)(cos(ang) * rope_attention_factor);
            m->rope_sin[(size_t)pos * half + i] = (float)(sin(ang) * rope_attention_factor);
        }
    }
}

static void rope_apply(float *x, const Model *m, int pos, int n_heads, int head_dim) {
    int half = head_dim / 2;
    const float *cs = m->rope_cos + (size_t)pos * half;
    const float *sn = m->rope_sin + (size_t)pos * half;
    for (int h = 0; h < n_heads; h++) {
        float *v = x + h * head_dim;
        for (int i = 0; i < half; i++) {
            float x1 = v[i], x2 = v[i + half];
            v[i] = x1 * cs[i] - x2 * sn[i];
            v[i + half] = x1 * sn[i] + x2 * cs[i];
        }
    }
}

static void ffn(Block *b, Model *m, const float *in, float *out) {
    matvec(b->w1, in, m->a);
    matvec(b->w3, in, m->b);
    silu_mul(m->a, m->b, m->pkg->ffn);
    matvec(b->w2, m->a, out);
}

/* forward d'un token ; remplit logits (vocab) */
static void forward_token(Model *m, int token, int pos, float *logits) {
    Package *pkg = m->pkg;
    const int d = pkg->d_model, nh = pkg->n_heads, nkv = pkg->n_kv, hd = pkg->head_dim;
    float *x = m->x, *h = m->h;

    embedding_row(m->emb, token, x);
    if (getenv("BCRUN_TRACE")) {
        fprintf(stderr, "[trace] emb x[0..3] = %.4f %.4f %.4f %.4f (rows=%d cols=%d dt=%d)\n",
                x[0], x[1], x[2], x[3], m->emb->rows, m->emb->cols, m->emb->dtype);
    }

    for (int slot = 0; slot < m->n_blocks_total; slot++) {
        Block *b = &m->blocks[slot];
        if (b->is_attn) {
            rmsnorm(h, x, b->norm, d, pkg->eps);
            matvec(b->q, h, m->q);
            matvec(b->k, h, m->k);
            matvec(b->v, h, m->v);

            /* QK-norm par tête */
            for (int hh = 0; hh < nh; hh++)
                rmsnorm(m->q + hh * hd, m->q + hh * hd, b->qnorm, hd, pkg->eps);
            for (int hh = 0; hh < nkv; hh++)
                rmsnorm(m->k + hh * hd, m->k + hh * hd, b->knorm, hd, pkg->eps);

            rope_apply(m->q, m, pos, nh, hd);
            rope_apply(m->k, m, pos, nkv, hd);

            /* écrire k,v dans le cache */
            int li = m->attn_layer_index[slot];
            uint16_t *kc = m->kv + ((size_t)(0 * m->n_attn + li) * m->ctx + pos) * nkv * hd;
            uint16_t *vc = m->kv + ((size_t)(1 * m->n_attn + li) * m->ctx + pos) * nkv * hd;
            for (int i = 0; i < nkv * hd; i++) {
                kc[i] = f2h(m->k[i]);
                vc[i] = f2h(m->v[i]);
            }

            int group = nh / nkv;
            for (int hh = 0; hh < nh; hh++) {
                int kvi = hh / group;
                float *qh = m->q + hh * hd;
                float maxs = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    const uint16_t *kt = m->kv +
                        ((size_t)(0 * m->n_attn + li) * m->ctx + t) * nkv * hd + kvi * hd;
                    float s = 0.0f;
                    for (int i = 0; i < hd; i++) s += qh[i] * h2f(kt[i]);
                    m->att[t] = s * (1.0f / sqrtf((float)hd));
                    if (m->att[t] > maxs) maxs = m->att[t];
                }
                float sum = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    m->att[t] = expf(m->att[t] - maxs);
                    sum += m->att[t];
                }
                float inv = 1.0f / sum;
                float *out = m->tmp + hh * hd;
                for (int i = 0; i < hd; i++) out[i] = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    float w = m->att[t] * inv;
                    const uint16_t *vt = m->kv +
                        ((size_t)(1 * m->n_attn + li) * m->ctx + t) * nkv * hd + kvi * hd;
                    for (int i = 0; i < hd; i++) out[i] += w * h2f(vt[i]);
                }
            }
            matvec(b->o, m->tmp, h);
            for (int i = 0; i < d; i++) x[i] += h[i];
        } else {
            rmsnorm(h, x, b->norm, d, pkg->eps);
            matvec(b->in_proj, h, m->gate);      /* 2d */
            float *a = m->gate, *g = m->gate + d;

            int ci = m->conv_slot[slot];
            float *cache = m->conv_cache + (size_t)ci * (pkg->kconv - 1) * d;
            /* concatène [cache(k-1), a(t)] et applique conv1d depthwise */
            const int k = pkg->kconv;
            for (int ch = 0; ch < d; ch++) {
                float acc = 0.0f;
                for (int j = 0; j < k - 1; j++)
                    acc += h2f(((const uint16_t *)b->conv->data)[ch * k + j]) * cache[j * d + ch];
                acc += h2f(((const uint16_t *)b->conv->data)[ch * k + (k - 1)]) * a[ch];
                m->tmp[ch] = acc;
            }
            /* met à jour le cache (décalage d'un cran) */
            for (int j = 0; j < k - 2; j++)
                for (int ch = 0; ch < d; ch++) cache[j * d + ch] = cache[(j + 1) * d + ch];
            for (int ch = 0; ch < d; ch++) cache[(k - 2) * d + ch] = a[ch];

            for (int ch = 0; ch < d; ch++) m->tmp[ch] *= g[ch];
            matvec(b->out_proj, m->tmp, h);
            for (int i = 0; i < d; i++) x[i] += h[i];
        }

        /* FFN */
        rmsnorm(h, x, b->ffn_norm, d, pkg->eps);
        ffn(b, m, h, m->tmp);
        for (int i = 0; i < d; i++) x[i] += m->tmp[i];
        if (getenv("BCRUN_TRACE"))
            fprintf(stderr, "[trace] block %d (%s) x[0..2] = %.5f %.5f %.5f\n",
                    slot, b->is_attn ? "attn" : "conv", x[0], x[1], x[2]);
    }

    rmsnorm(h, x, m->norm_f, d, pkg->eps);
    matvec(m->emb, h, logits);        /* lm_head lié (int8 ou fp16) */
}

/* ------------------------------------------------------------------ */
/* génération                                                          */
/* ------------------------------------------------------------------ */
static uint32_t rng_state = 123456789u;
static float rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (rng_state & 0xffffff) / (float)0x1000000;
}

static int sample(float *logits, int vocab, float temp, int topk) {
    if (temp <= 1e-6f) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    if (topk > 0 && topk < vocab) {
        int *idx = xmalloc(sizeof(int) * vocab);
        for (int i = 0; i < vocab; i++) idx[i] = i;
        /* tri partiel : sélection simple des top-k */
        for (int i = 0; i < topk; i++) {
            int m = i;
            for (int j = i + 1; j < vocab; j++)
                if (logits[idx[j]] > logits[idx[m]]) m = j;
            int tmp = idx[i]; idx[i] = idx[m]; idx[m] = tmp;
        }
        float maxl = logits[idx[0]], sum = 0.0f;
        for (int i = 0; i < topk; i++) { logits[idx[i]] = expf((logits[idx[i]] - maxl) / temp); sum += logits[idx[i]]; }
        float r = rnd() * sum;
        int out = idx[0];
        for (int i = 0; i < topk; i++) { r -= logits[idx[i]]; if (r <= 0) { out = idx[i]; break; } }
        free(idx);
        return out;
    }
    float maxl = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > maxl) maxl = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) { logits[i] = expf((logits[i] - maxl) / temp); sum += logits[i]; }
    float r = rnd() * sum;
    for (int i = 0; i < vocab; i++) { r -= logits[i]; if (r <= 0) return i; }
    return vocab - 1;
}

/* ------------------------------------------------------------------ */
static void append_id(int **ids, size_t *n_ids, size_t *capacity, int value) {
    if (*n_ids == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 256;
        if (next > SIZE_MAX / sizeof(int)) die("trop d'IDs en entrée");
        int *grown = realloc(*ids, next * sizeof(int));
        if (!grown) die("allocation des IDs impossible");
        *ids = grown;
        *capacity = next;
    }
    (*ids)[(*n_ids)++] = value;
}

int main(int argc, char **argv) {
    const char *pkg_path = NULL, *dump = NULL;
    int *ids = NULL;
    size_t n_ids = 0, ids_capacity = 0;
    int max_new = 8, ctx = 4096, topk = 0, mem_report = 0, quiet = 0, debug = 0;
    float temp = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pkg") && i + 1 < argc) pkg_path = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) {
            char *s = argv[++i];
            while (*s) {
                char *end = NULL;
                long value = strtol(s, &end, 10);
                if (end == s || value < 0 || value > INT_MAX)
                    die("liste --ids invalide");
                append_id(&ids, &n_ids, &ids_capacity, (int)value);
                s = end;
                if (*s == ',') s++;
                else if (*s != '\0') die("séparateur invalide dans --ids");
            }
        }
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--context") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--mem-report")) mem_report = 1;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--debug")) debug = 1;
        else { fprintf(stderr, "option inconnue : %s\n", argv[i]); return 1; }
    }
    if (!pkg_path) die("--pkg requis");
    if (n_ids == 0) append_id(&ids, &n_ids, &ids_capacity, 1);
    if (ctx <= 0) die("--context doit être positif");
    if (max_new < 0 || max_new > 4096) die("--max-new doit être compris entre 0 et 4096");

    double t0 = now_s();
    double rss0 = rss_mb();
    Package pkg;
    load_package(&pkg, pkg_path);
    double rss_after_map = rss_mb();

    if (debug) {
        for (int i = 0; i < pkg.n_tensors; i++) {
            Tensor *t = &pkg.tensors[i];
            printf("  %-40s dt=%d rows=%d cols=%d pc=%d shape=%d,%d,%d\n",
                   t->name, t->dtype, t->rows, t->cols, t->packed_cols,
                   t->shape[0], t->shape[1], t->shape[2]);
        }
    }
    if (ctx > pkg.max_seq) ctx = pkg.max_seq;
    if (n_ids > (size_t)ctx) die("prompt plus long que le contexte demandé");
    for (size_t i = 0; i < n_ids; i++)
        if (ids[i] >= pkg.vocab) die("ID de prompt hors vocabulaire");
    Model m;
    model_init(&m, &pkg, ctx);
    double rss_after_alloc = rss_mb();

    float *logits = xmalloc(sizeof(float) * pkg.vocab);

    double tf0 = now_s();
    for (size_t i = 0; i < n_ids; i++) forward_token(&m, ids[i], (int)i, logits);
    double tf1 = now_s();

    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f) die("impossible d'écrire les logits");
        fwrite(logits, sizeof(float), pkg.vocab, f);
        fclose(f);
    }

    if (!quiet) {
        printf("modèle          : %d couches, d=%d, %d têtes / %d KV, ffn=%d, vocab=%d\n",
               pkg.n_layers, pkg.d_model, pkg.n_heads, pkg.n_kv, pkg.ffn, pkg.vocab);
        printf("contexte        : %d (max %d), boucle ×%d, conv k=%d, 1 attention / %d blocs\n",
               ctx, pkg.max_seq, pkg.loop, pkg.kconv, pkg.attn_every);
        printf("paquet          : %.1f Mo (mmap, sans copie)\n",
               pkg.map_size / (1024.0 * 1024.0));
        printf("prefill         : %zu tokens en %.2f s (%.1f tok/s)\n",
               n_ids, tf1 - tf0, (double)n_ids / (tf1 - tf0 + 1e-9));
    }

    int generated[4096];
    int n_gen = 0;
    double tg0 = now_s();
    int pos = (int)n_ids;
    for (int g = 0; g < max_new && pos < ctx; g++) {
        int next = sample(logits, pkg.vocab, temp, topk);
        generated[n_gen++] = next;
        forward_token(&m, next, pos, logits);
        pos++;
    }
    double tg1 = now_s();

    if (!quiet) {
        printf("génération      : %d tokens en %.2f s (%.1f tok/s)\n",
               n_gen, tg1 - tg0, n_gen / (tg1 - tg0 + 1e-9));
        printf("tokens générés  :");
        for (int i = 0; i < n_gen; i++) printf(" %d", generated[i]);
        printf("\n");
    }

    if (mem_report) {
        double peak = rss_mb();
        printf("\n--- RAM (RSS, pages réellement touchées) ---\n");
        printf("avant chargement        : %7.1f Mo\n", rss0);
        printf("après mmap du paquet    : %7.1f Mo\n", rss_after_map);
        printf("après cache KV + conv   : %7.1f Mo\n", rss_after_alloc);
        printf("pic pendant l'inférence : %7.1f Mo\n", peak);
        printf("total depuis le départ  : %7.2f s\n", now_s() - t0);
        printf("détail cache KV         : %7.1f Mo (%d couches d'attention × %d tokens)\n",
               (double)m.n_attn * 2 * pkg.n_kv * pkg.head_dim * ctx * 2 / (1024 * 1024),
               m.n_attn, ctx);
    }
    free(logits);
    free(ids);
    return 0;
}
