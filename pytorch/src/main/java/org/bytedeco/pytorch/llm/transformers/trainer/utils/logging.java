/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.trainer.utils;

import java.util.logging.Logger;

/**
 * Logging utilities mirroring Python's {@code transformers.utils.logging}.
 *
 * <p>Wraps {@link Logger} and respects the
 * {@code TRANSFORMERS_VERBOSITY} environment variable.
 */
public final class logging {

    private static volatile String current_verbosity;

    private logging() {}

    /**
     * Get a {@link Logger} for the given name (mirrors Python's {@code logging.getLogger}).
     *
     * @param name logger name
     * @return configured {@link Logger}
     */
    public static Logger get_logger(String name) {
        Logger logger = Logger.getLogger(name);
        String verbosity = current_verbosity;
        if (verbosity == null) {
            verbosity = System.getenv("TRANSFORMERS_VERBOSITY");
            if (verbosity == null) verbosity = "warning";
            current_verbosity = verbosity;
        }
        switch (verbosity.toLowerCase()) {
            case "debug"   -> logger.setParent(Logger.getGlobal());
            case "info"    -> logger.setParent(Logger.getLogger("global"));
            case "warning" -> logger.setParent(Logger.getLogger("warning"));
            case "error"   -> logger.setParent(Logger.getLogger("severe"));
            default        -> logger.setParent(Logger.getLogger("global"));
        }
        return logger;
    }

    /**
     * Set the global transformers verbosity for all subsequent loggers.
     *
     * @param level "debug", "info", "warning", or "error"
     */
    public static void set_verbosity(String level) {
        current_verbosity = level;
    }
}
