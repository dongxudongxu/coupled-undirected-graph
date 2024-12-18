#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import graphscope
import pytest
from graphscope.dataset import load_p2p_network
from graphscope.framework.app import load_app


@pytest.fixture(scope="module")
def session():
    graphscope.set_option(show_log=True)
    sess = graphscope.session(cluster_type="hosts")
    yield sess
    sess.close()


@pytest.fixture(scope="module")
def p2p_property_graph(session):
    graph = load_p2p_network(session)
    yield graph
    del graph


@pytest.fixture(scope="module")
def p2p_projected_graph(p2p_property_graph):
    graph = p2p_property_graph._project_to_simple()
    yield graph
    del graph


def test_vertex_degree_app(name, p2p_projected_graph):
    app = load_app(os.path.join(os.getcwd(), "{0}.gar".format(name)))
    ctx = app(p2p_projected_graph, 10)
    print(ctx.to_numpy("r"))
