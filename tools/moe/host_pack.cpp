// Offline, host-only Hexagon tiled-pack shim.
//
// A100/x86 hosts cannot load the Android Hexagon backend, but the tiled layout
// converter itself is plain C++ (ggml/src/ggml-hexagon/htp-repack.cpp).  This
// shim exposes the same `ggml_backend_pack_moe_expert` entry point as the real
// backend so tools/moe/pack_hexagon.py can generate deployment assets on a
// build host.  It intentionally exposes no devices and is not a runtime
// backend.
//
// Build (from the repository root, after configuring a build directory):
//   g++ -O2 -fPIC -shared -std=c++17 \
//       -I ggml/include -I ggml/src -I ggml/src/ggml-hexagon \
//       -I ggml/src/ggml-hexagon/htp \
//       tools/moe/host_pack.cpp ggml/src/ggml-hexagon/htp-repack.cpp \
//       -o "$BUILD_DIR/bin/libggml-hexagon.so" \
//       -L "$BUILD_DIR/bin" -lggml-base -Wl,-rpath,'$ORIGIN'
//
// The Q5_K path is the same code the device backend runs: htp_is_tiled_weight_type()
// accepts type 13 and htp_repack_tensor_tiled() uses HTP_TENSOR_TILE_SIZE_Q5_K (768).

#include "ggml-backend-impl.h"
#include "htp-repack.h"

#include <cstdint>
#include <cstring>

namespace {
int64_t round_up(int64_t n, int64_t m) { return ((n + m - 1) / m) * m; }

size_t pack_moe_expert(void * dst, const void * src, int type, int64_t ne0, int64_t ne1) {
    if (ne0 <= 0 || ne1 <= 0 || !htp_is_tiled_weight_type((ggml_type) type)) return 0;
    const int64_t ne0_pad = round_up(ne0, 32);
    const int64_t ne1_pad = round_up(ne1, 32);
    const size_t bytes = htp_tiled_row_size((ggml_type) type, ne0_pad) * ne1_pad;
    if (dst && src) {
        ggml_tensor tensor = {};
        tensor.type = (ggml_type) type;
        tensor.ne[0] = ne0; tensor.ne[1] = ne1; tensor.ne[2] = tensor.ne[3] = 1;
        tensor.data = dst;
        htp_repack_tensor_tiled(&tensor, src, ggml_row_size(tensor.type, ne0) * ne1);
    }
    return bytes;
}

const char * reg_name(ggml_backend_reg_t) { return "hexagon-host-pack"; }
size_t reg_device_count(ggml_backend_reg_t) { return 0; }
ggml_backend_dev_t reg_device(ggml_backend_reg_t, size_t) { return nullptr; }
void * reg_proc(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_pack_moe_expert") == 0) return (void *) pack_moe_expert;
    return nullptr;
}
ggml_backend_reg reg = {
    GGML_BACKEND_API_VERSION,
    { reg_name, reg_device_count, reg_device, reg_proc },
    nullptr,
};
}

extern "C" ggml_backend_reg_t ggml_backend_hexagon_reg(void) { return &reg; }
