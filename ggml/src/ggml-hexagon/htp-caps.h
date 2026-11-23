#ifndef HTP_CAPS_H
#define HTP_CAPS_H

// Capabilities advertised by the HTP skel through htp_iface_capabilities().
// Add new capability bits here; old skels report no capabilities.
//
// Bit 0 is HTP_CAP_BIT_Q3K_COMPACT (htp/htp-ops.h, device side). Keep this
// namespace shared by host and skel: a bit must mean the same thing on both
// sides or a host that supports one feature enables another.
enum htp_capability {
    HTP_CAP_ROWNORM = (1u << 1),
};

#endif // HTP_CAPS_H
