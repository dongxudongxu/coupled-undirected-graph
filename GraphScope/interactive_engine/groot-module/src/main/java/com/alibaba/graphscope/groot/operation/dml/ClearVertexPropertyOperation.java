/**
 * Copyright 2020 Alibaba Group Holding Limited.
 *
 * <p>Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain a copy of the License at
 *
 * <p>http://www.apache.org/licenses/LICENSE-2.0
 *
 * <p>Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.alibaba.graphscope.groot.operation.dml;

import com.alibaba.graphscope.groot.common.schema.wrapper.LabelId;
import com.alibaba.graphscope.groot.operation.Operation;
import com.alibaba.graphscope.groot.operation.OperationType;
import com.alibaba.graphscope.groot.operation.VertexId;
import com.alibaba.graphscope.proto.groot.DataOperationPb;
import com.google.protobuf.ByteString;

import java.util.List;

public class ClearVertexPropertyOperation extends Operation {

    private final VertexId vertexId;
    private final LabelId labelId;
    private final List<Integer> properties;

    public ClearVertexPropertyOperation(
            VertexId vertexId, LabelId labelId, List<Integer> properties) {
        super(OperationType.CLEAR_VERTEX_PROPERTIES);
        this.vertexId = vertexId;
        this.labelId = labelId;
        this.properties = properties;
    }

    @Override
    protected long getPartitionKey() {
        return vertexId.getId();
    }

    @Override
    protected ByteString getBytes() {
        DataOperationPb.Builder builder = DataOperationPb.newBuilder();
        builder.setKeyBlob(vertexId.toProto().toByteString());
        builder.setLocationBlob(labelId.toProto().toByteString());
        builder.addAllPropIds(properties);
        return builder.build().toByteString();
    }
}
