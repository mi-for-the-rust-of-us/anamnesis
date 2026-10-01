// SPDX-License-Identifier: MIT OR Apache-2.0
//
// ggml_ref: quantise or dequantise through ggml's OWN type traits.
//
//   ggml_ref quantize   <ggml_type> <n_elements>   f32 in on stdin, blocks out
//   ggml_ref dequantize <ggml_type> <n_elements>   blocks in on stdin, f32 out
//
// Exists for the types `gguf-py` does not implement (Phase 7.10: it has no
// quantiser for NVFP4, Q1_0 or Q2_0, and no dequantiser for Q1_0 or Q2_0).
// The project's fixture rule is that a golden comes from the canonical
// library's own code, never from a reimplementation, and for these types the
// canonical code is ggml's C: `from_float_ref` is `quantize_row_*_ref` and
// `to_float` is `dequantize_row_*`, straight out of `ggml-quants.c`.
//
// Built against llama.cpp 37b53fd by build.sh; driven by generate_gguf.py.

#include "ggml.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *msg) {
    fprintf(stderr, "ggml_ref: %s\n", msg);
    return 1;
}

static int read_exact(void *buf, size_t len) {
    return fread(buf, 1, len, stdin) == len && fgetc(stdin) == EOF;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return fail("usage: ggml_ref quantize|dequantize <ggml_type> <n_elements>");
    }
    const int type = atoi(argv[2]);
    const long long n = atoll(argv[3]);
    if (type < 0 || type >= GGML_TYPE_COUNT || n <= 0) {
        return fail("bad type or element count");
    }

    // Initialise ggml once, so any table a conversion relies on is set up
    // exactly as it is in llama.cpp itself.
    struct ggml_init_params params = {1024, NULL, true};
    ggml_free(ggml_init(params));

    const struct ggml_type_traits *traits = ggml_get_type_traits((enum ggml_type) type);
    const long long block = traits->blck_size;
    if (block <= 0 || n % block != 0) {
        return fail("element count is not a whole number of blocks");
    }
    const size_t raw_len = (size_t) (n / block) * traits->type_size;
    const size_t f32_len = (size_t) n * sizeof(float);

    if (strcmp(argv[1], "quantize") == 0) {
        if (traits->from_float_ref == NULL) {
            return fail("type has no reference quantiser");
        }
        float *x = malloc(f32_len);
        void *y = malloc(raw_len);
        if (x == NULL || y == NULL) return fail("out of memory");
        if (!read_exact(x, f32_len)) return fail("stdin is not n_elements f32 values");
        traits->from_float_ref(x, y, n);
        fwrite(y, 1, raw_len, stdout);
    } else if (strcmp(argv[1], "dequantize") == 0) {
        if (traits->to_float == NULL) {
            return fail("type has no dequantiser");
        }
        void *y = malloc(raw_len);
        float *x = malloc(f32_len);
        if (x == NULL || y == NULL) return fail("out of memory");
        if (!read_exact(y, raw_len)) return fail("stdin is not n_elements' worth of blocks");
        traits->to_float(y, x, n);
        fwrite(x, 1, f32_len, stdout);
    } else {
        return fail("mode must be quantize or dequantize");
    }
    return fflush(stdout) == 0 ? 0 : fail("write failed");
}
