/*
 * Copyright 2020 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.alibaba.graphscope.common.config;

public class GraphConfig {
    public static final Config<String> GRAPH_SCHEMA = Config.stringConfig("graph.schema", ".");
    public static final Config<String> GRAPH_STATISTICS =
            Config.stringConfig("graph.statistics", "");
    public static final Config<String> GRAPH_STORE = Config.stringConfig("graph.store", "exp");

    @Deprecated
    public static final Config<String> GRAPH_STORED_PROCEDURES =
            Config.stringConfig("graph.stored.procedures", "");

    @Deprecated
    public static final Config<String> GRAPH_STORED_PROCEDURES_ENABLE_LISTS =
            Config.stringConfig("graph.stored.procedures.enable.lists", "");

    // denote stored procedures in yaml format, refer to test resource file
    // 'config/modern/graph.yaml' for more info about
    // the format
    public static final Config<String> GRAPH_STORED_PROCEDURES_YAML =
            Config.stringConfig("graph.stored.procedures.yaml", "");
}
