/**
 * @file sha256.c
 * @brief SHA-256 — Pure C Implementation
 *
 * Portable SHA-256 with zero external dependencies.
 * Follows FIPS 180-4 specification exactly.
 *
 * This replaces the C++ SHA-256 fallback in integrity.hpp with
 * a cleaner pure-C implementation that can be used standalone
 * or linked into the C++/pybind11 layer.
 *
 * Tier-0 Protocol: integrity verification infrastructure.
 */

#include "umcp_c/sha256.h"

#include <stdio.h>
#include <string.h>

/* ─── Constants (FIPS 180-4 §4.2.2) ────────────────────────────── */

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* ─── Bit operations ────────────────────────────────────────────── */

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2)  ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6)  ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7)  ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

/* ─── Process one 64-byte block ─────────────────────────────────── */

static void sha256_transform(uint32_t state[8], const uint8_t block[64])
{
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

    /* Prepare message schedule (FIPS 180-4 §6.2.2) */
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i * 4 + 0] << 24)
             | ((uint32_t)block[i * 4 + 1] << 16)
             | ((uint32_t)block[i * 4 + 2] <<  8)
             | ((uint32_t)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
    }

    /* Initialize working variables */
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    /* 64 rounds */
    for (int i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* ─── Streaming API ─────────────────────────────────────────────── */

void umcp_sha256_init(umcp_sha256_ctx_t *ctx)
{
    /* FIPS 180-4 §5.3.3: Initial hash values */
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->total_len = 0;
    ctx->block_len = 0;
}

void umcp_sha256_update(umcp_sha256_ctx_t *ctx,
                        const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;

    ctx->total_len += len;

    /* Fill partial block */
    if (ctx->block_len > 0) {
        size_t need = UMCP_SHA256_BLOCK_SIZE - ctx->block_len;
        if (len < need) {
            memcpy(ctx->block + ctx->block_len, p, len);
            ctx->block_len += len;
            return;
        }
        memcpy(ctx->block + ctx->block_len, p, need);
        sha256_transform(ctx->state, ctx->block);
        p   += need;
        len -= need;
        ctx->block_len = 0;
    }

    /* Process full blocks */
    while (len >= UMCP_SHA256_BLOCK_SIZE) {
        sha256_transform(ctx->state, p);
        p   += UMCP_SHA256_BLOCK_SIZE;
        len -= UMCP_SHA256_BLOCK_SIZE;
    }

    /* Store remainder */
    if (len > 0) {
        memcpy(ctx->block, p, len);
        ctx->block_len = len;
    }
}

void umcp_sha256_final(umcp_sha256_ctx_t *ctx, uint8_t digest[32])
{
    uint64_t total_bits = ctx->total_len * 8;

    /* Padding: append 1 bit, then zeros */
    ctx->block[ctx->block_len++] = 0x80;

    if (ctx->block_len > 56) {
        memset(ctx->block + ctx->block_len, 0,
               UMCP_SHA256_BLOCK_SIZE - ctx->block_len);
        sha256_transform(ctx->state, ctx->block);
        ctx->block_len = 0;
    }
    memset(ctx->block + ctx->block_len, 0,
           56 - ctx->block_len);

    /* Append message length in bits (big-endian) */
    for (int i = 7; i >= 0; --i) {
        ctx->block[56 + (7 - i)] = (uint8_t)(total_bits >> (i * 8));
    }
    sha256_transform(ctx->state, ctx->block);

    /* Produce digest (big-endian) */
    for (int i = 0; i < 8; ++i) {
        digest[i * 4 + 0] = (uint8_t)(ctx->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(ctx->state[i] >>  8);
        digest[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

/* ─── Convenience functions ─────────────────────────────────────── */

static void digest_to_hex(const uint8_t digest[32],
                          char hex_out[UMCP_SHA256_HEX_SIZE])
{
    static const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        hex_out[i * 2 + 0] = hex_chars[(digest[i] >> 4) & 0x0f];
        hex_out[i * 2 + 1] = hex_chars[digest[i] & 0x0f];
    }
    hex_out[64] = '\0';
}

void umcp_sha256_hex(const void *data, size_t len,
                     char hex_out[UMCP_SHA256_HEX_SIZE])
{
    umcp_sha256_ctx_t ctx;
    uint8_t digest[UMCP_SHA256_DIGEST_SIZE];

    umcp_sha256_init(&ctx);
    umcp_sha256_update(&ctx, data, len);
    umcp_sha256_final(&ctx, digest);
    digest_to_hex(digest, hex_out);
}

int umcp_sha256_file(const char *filepath,
                     char hex_out[UMCP_SHA256_HEX_SIZE])
{
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;

    umcp_sha256_ctx_t ctx;
    umcp_sha256_init(&ctx);

    uint8_t buf[262144];  /* 256 KB read buffer for I/O throughput */
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        umcp_sha256_update(&ctx, buf, n);
    }
    fclose(fp);

    uint8_t digest[UMCP_SHA256_DIGEST_SIZE];
    umcp_sha256_final(&ctx, digest);
    digest_to_hex(digest, hex_out);

    return 0;
}
