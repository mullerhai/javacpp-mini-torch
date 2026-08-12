/*
 * IntegrityChecker -- bulk-verify that signable entries stored at-rest in a
 * CacheBackend have not been tampered with.
 *
 * <p>Walks the backend (or a sliced subset), re-computes the HMAC and
 * reports mismatches. Intended to run as a periodic cron job in a
 * security-ops pipeline.
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;

public final class IntegrityChecker {

    private final IntegritySigner signer;

    public IntegrityChecker(IntegritySigner signer) {
        this.signer = signer;
    }

    public Report run(CacheBackend backend, Collection<CacheKey> keys) {
        long verified = 0, tampered = 0, missing = 0;
        List<CacheKey> tamperedKeys = new ArrayList<>();
        for (CacheKey k : keys) {
            Optional<CacheValue<Object>> v = backend.get(k);
            if (!v.isPresent()) { missing++; continue; }
            String token = v.get().tag("integrity");
            if (token == null) { missing++; continue; }
            if (signer.verify(k, v.get(), token)) {
                verified++;
            } else {
                tampered++;
                tamperedKeys.add(k);
            }
        }
        return new Report(verified, tampered, missing, tamperedKeys);
    }

    public static final class Report {
        public final long verified, tampered, missing;
        public final List<CacheKey> tamperedKeys;

        public Report(long verified, long tampered, long missing, List<CacheKey> tamperedKeys) {
            this.verified = verified;
            this.tampered = tampered;
            this.missing = missing;
            this.tamperedKeys = tamperedKeys;
        }

        public boolean isClean() { return tampered == 0; }

        @Override
        public String toString() {
            return "IntegrityReport{verified=" + verified + ", tampered=" + tampered
                    + ", missing=" + missing + "}";
        }
    }
}
