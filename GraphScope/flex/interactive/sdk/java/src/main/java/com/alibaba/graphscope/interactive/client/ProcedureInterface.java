/*
 * Copyright 2022 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  	http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package com.alibaba.graphscope.interactive.client;

import com.alibaba.graphscope.gaia.proto.IrResult;
import com.alibaba.graphscope.interactive.client.common.Result;
import com.alibaba.graphscope.interactive.openapi.model.*;

import java.util.List;

/**
 * All APIs about procedure management.
 * TODO(zhanglei): differ between ProcedureRequest and Procedure
 */
public interface ProcedureInterface {
    Result<CreateProcedureResponse> createProcedure(
            String graphId, CreateProcedureRequest procedure);

    Result<String> deleteProcedure(String graphId, String procedureName);

    Result<GetProcedureResponse> getProcedure(String graphId, String procedureName);

    Result<List<GetProcedureResponse>> listProcedures(String graphId);

    Result<String> updateProcedure(
            String graphId, String procedureId, UpdateProcedureRequest procedure);

    Result<IrResult.CollectiveResults> callProcedure(String graphId, QueryRequest request);

    Result<String> callProcedureRaw(String graphId, String request);
}
