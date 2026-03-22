/**
 * @file sha256.h
 * @brief SHA-256 — Pure C Implementation (Tier-0 Protocol)
 *
 * Portable SHA-256 with no external dependencies.
 * Used for integrity verification of tracked files.
 *
 * Design:
 *   - Streaming API (init/update/final) for arbitrary-length data
 *   - File convenience function with configurable buffer size
 *   - No heap allocation — all state in caller-provided struct
 *   - C99 with fixed-width integer types
 */

#ifndef UMCP_C_SHA256_H
#define UMCP_C_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UMCP_SHA256_DIGEST_SIZE  32
#define UMCP_SHA256_BLOCK_SIZE   64
#define UMCP_SHA256_HEX_SIZE     65  /* 64 hex chars + NUL */

/* ─── Streaming context ─────────────────────────────────────────── */

typedef struct {
    uint32_t state[8];
    uint64_t total_len;
    uint8_t  block[UMCP_SHA256_BLOCK_SIZE];
    size_t   block_len;
} umcp_sha256_ctx_t;

/**
 * Initialize a SHA-256 context.
 */
void umcp_sha256_init(umcp_sha256_ctx_t *ctx);

/**
 * Feed data into the hash.
 *
 * @param ctx  Initialized context
 * @param data Input bytes
 * @param len  Number of bytes
 */
void umcp_sha256_update(umcp_sha256_ctx_t *ctx,
                        const void *data, size_t len);

/**
 * Finalize and produce the 32-byte digest.
 *
 * @param ctx    Context (consumed — must re-init for reuse)
 * @param digest Output buffer (at least UMCP_SHA256_DIGEST_SIZE bytes)
 */
void umcp_sha256_final(umcp_sha256_ctx_t *ctx, uint8_t digest[32]);

/* ─── Convenience functions ─────────────────────────────────────── */

/**
 * Hash a memory buffer and produce a hex string.
 *
 * @param data    Input buffer
 * @param len     Length in bytes
 * @param hex_out Output hex string (at least UMCP_SHA256_HEX_SIZE bytes)
 */
void umcp_sha256_hex(const void *data, size_t len,
                     char hex_out[UMCP_SHA256_HEX_SIZE]);

/**
 * Hash a file and produce a hex string.
 *
 * @param filepath  Path to file
 * @param hex_out   Output hex string (at least UMCP_SHA256_HEX_SIZE bytes)
 * @return 0 on success, -1 if file cannot be opened
 */
int umcp_sha256_file(const char *filepath,
                     char hex_out[UMCP_SHA256_HEX_SIZE]);

#ifdef __cplusplus
}
#endif

#endif /* UMCP_C_SHA256_H */
