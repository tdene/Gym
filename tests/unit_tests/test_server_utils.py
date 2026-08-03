# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import socket
import time
from unittest.mock import AsyncMock, MagicMock

from aiohttp import ClientOSError
from pytest import MonkeyPatch, mark, raises

import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
)
from nemo_gym.server_utils import (
    BaseServer,
    BaseServerConfig,
    ConnectionError,
    DictConfig,
    GlobalAIOHTTPAsyncClientConfig,
    HeadServer,
    ServerClient,
    SimpleServer,
    _make_keepalive_socket_factory,
    initialize_ray,
)


_TCP_KEEPALIVE_TEST_IDLE = 42
_TCP_KEEPALIVE_TEST_INTERVAL = 7
_TCP_KEEPALIVE_TEST_PROBES = 2
_TEST_ADDR_INFO = (
    socket.AF_INET,
    socket.SOCK_STREAM,
    socket.IPPROTO_TCP,
    "",
    ("203.0.113.1", 443),
)


class TestServerUtils:
    def test_global_aiohttp_client_request_debug_enabled(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)
        assert not nemo_gym.server_utils.is_global_aiohttp_client_request_debug_enabled()

        monkeypatch.setattr(nemo_gym.server_utils, "_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", True)
        assert nemo_gym.server_utils.is_global_aiohttp_client_request_debug_enabled()

    def test_ServerClient_load_head_server_config(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)
        actual_config = ServerClient.load_head_server_config()
        assert actual_config.host == ""
        assert actual_config.port == 0

    def test_ServerClient_load_from_global_config(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        httpx_client_mock = MagicMock()
        httpx_response_mock = MagicMock()
        httpx_client_mock.return_value = httpx_response_mock
        httpx_response_mock.content = b'"a: 2"'
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", httpx_client_mock)

        actual_client = ServerClient.load_from_global_config()
        assert {"a": 2} == actual_client.global_config_dict

    def test_ServerClient_load_from_global_config_propogate_ConnectionError(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig(
            {
                "head_server": {
                    "host": "",
                    "port": 0,
                }
            }
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        httpx_client_mock = MagicMock()
        httpx_client_mock.side_effect = ConnectionError
        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", httpx_client_mock)

        with raises(ValueError):
            ServerClient.load_from_global_config()

    async def test_ServerClient_get_post_sanity(self, monkeypatch: MonkeyPatch) -> None:
        server_client = ServerClient(
            head_server_config=BaseServerConfig(host="abcdef", port=12345),
            global_config_dict=DictConfig(
                {
                    "my_server": {
                        "a": {
                            "b": {
                                "host": "xyz",
                                "port": 54321,
                            }
                        }
                    }
                }
            ),
        )

        httpx_client_mock = MagicMock()
        httpx_client_request_mock = AsyncMock()
        httpx_client_request_mock.return_value = "my mock response"
        httpx_client_mock.return_value.request = httpx_client_request_mock
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", httpx_client_mock)

        actual_response = await server_client.get(
            server_name="my_server",
            url_path="blah blah",
        )
        assert "my mock response" == actual_response

        actual_response = await server_client.post(
            server_name="my_server",
            url_path="blah blah",
        )
        assert "my mock response" == actual_response

    def test_BaseServer_load_config_from_global_config(self, monkeypatch: MonkeyPatch) -> None:
        # Clear any lingering env vars.
        monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, "my_server")

        global_config_dict = DictConfig(
            {"my_server": {"a": {"b": {"host": "", "port": 0, "entrypoint": "my entrypoint"}}}}
        )
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        actual_config = BaseServer.load_config_from_global_config()
        assert "" == actual_config.host
        assert 0 == actual_config.port
        assert "my entrypoint" == actual_config.entrypoint

    def test_HeadServer_setup_webserver_sanity(self) -> None:
        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        head_server.setup_webserver()

    async def test_HeadServer_global_config_dict_yaml(self, monkeypatch: MonkeyPatch) -> None:
        global_config_dict = DictConfig({"a": 2})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        head_server = HeadServer(config=BaseServerConfig(host="", port=0))
        resp = await head_server.global_config_dict_yaml()

        assert "a: 2\n" == resp

    def _mock_ray_return_value(self, monkeypatch: MonkeyPatch, return_value: bool) -> MagicMock:
        ray_is_initialized_mock = MagicMock()
        ray_is_initialized_mock.return_value = return_value
        monkeypatch.setattr(nemo_gym.server_utils.ray, "is_initialized", ray_is_initialized_mock)
        return ray_is_initialized_mock

    def _mock_ray_init(self, monkeypatch: MonkeyPatch) -> MagicMock:
        ray_init_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils.ray, "init", ray_init_mock)
        return ray_init_mock

    def test_initialize_ray_already_initialized(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, True)

        get_global_config_dict_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        initialize_ray()

        ray_is_initialized_mock.assert_called_once()
        get_global_config_dict_mock.assert_not_called()

    def test_initialize_ray_with_address(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, False)

        ray_init_mock = self._mock_ray_init(monkeypatch)

        # Mock global config dict with ray_head_node_address
        global_config_dict = DictConfig({"ray_head_node_address": "ray://test-address:10001"})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        initialize_ray()

        ray_is_initialized_mock.assert_called_once()
        get_global_config_dict_mock.assert_called_once()
        ray_init_mock.assert_called_once_with(address="ray://test-address:10001", ignore_reinit_error=True)

    def test_initialize_ray_without_address(self, monkeypatch: MonkeyPatch) -> None:
        ray_is_initialized_mock = self._mock_ray_return_value(monkeypatch, False)

        ray_init_mock = self._mock_ray_init(monkeypatch)

        ray_runtime_context_mock = MagicMock()
        ray_runtime_context_mock.gcs_address = "ray://mock-address:10001"
        ray_get_runtime_context_mock = MagicMock()
        ray_get_runtime_context_mock.return_value = ray_runtime_context_mock
        monkeypatch.setattr(nemo_gym.server_utils.ray, "get_runtime_context", ray_get_runtime_context_mock)

        # Mock global config dict without ray_head_node_address
        global_config_dict = DictConfig({"k": "v"})
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = global_config_dict
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        initialize_ray()

        ray_is_initialized_mock.assert_called_once()
        get_global_config_dict_mock.assert_called_once()
        ray_init_mock.assert_called_once_with(ignore_reinit_error=True)
        ray_get_runtime_context_mock.assert_called_once()

    def test_keepalive_socket_factory_sets_keepalive_sockopts(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)

        factory = _make_keepalive_socket_factory(
            idle_seconds=_TCP_KEEPALIVE_TEST_IDLE,
            interval_seconds=_TCP_KEEPALIVE_TEST_INTERVAL,
            probes=_TCP_KEEPALIVE_TEST_PROBES,
        )
        result = factory(_TEST_ADDR_INFO)

        assert result is mock_sock
        socket_ctor_mock.assert_called_once_with(
            family=_TEST_ADDR_INFO[0], type=_TEST_ADDR_INFO[1], proto=_TEST_ADDR_INFO[2]
        )
        mock_sock.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt_name, opt_value in (
            ("TCP_KEEPIDLE", _TCP_KEEPALIVE_TEST_IDLE),
            ("TCP_KEEPINTVL", _TCP_KEEPALIVE_TEST_INTERVAL),
            ("TCP_KEEPCNT", _TCP_KEEPALIVE_TEST_PROBES),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, opt, opt_value)

    def test_keepalive_socket_factory_skips_missing_platform_sockopts(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)
        for opt_name in ("TCP_KEEPIDLE", "TCP_KEEPINTVL", "TCP_KEEPCNT"):
            monkeypatch.delattr(socket, opt_name, raising=False)

        factory = _make_keepalive_socket_factory(
            idle_seconds=_TCP_KEEPALIVE_TEST_IDLE,
            interval_seconds=_TCP_KEEPALIVE_TEST_INTERVAL,
            probes=_TCP_KEEPALIVE_TEST_PROBES,
        )
        factory(_TEST_ADDR_INFO)

        mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def test_GlobalAIOHTTPAsyncClientConfig_keepalive_defaults(self) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig()
        assert cfg.global_aiohttp_tcp_keepalive_idle_seconds == 60
        assert cfg.global_aiohttp_tcp_keepalive_interval_seconds == 10
        assert cfg.global_aiohttp_tcp_keepalive_probes == 3

    def test_keepalive_socket_factory_uses_configured_values(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)

        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_tcp_keepalive_idle_seconds=123,
            global_aiohttp_tcp_keepalive_interval_seconds=45,
            global_aiohttp_tcp_keepalive_probes=6,
        )
        factory = _make_keepalive_socket_factory(
            idle_seconds=cfg.global_aiohttp_tcp_keepalive_idle_seconds,
            interval_seconds=cfg.global_aiohttp_tcp_keepalive_interval_seconds,
            probes=cfg.global_aiohttp_tcp_keepalive_probes,
        )
        factory(_TEST_ADDR_INFO)

        for opt_name, opt_value in (
            ("TCP_KEEPIDLE", 123),
            ("TCP_KEEPINTVL", 45),
            ("TCP_KEEPCNT", 6),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, opt, opt_value)

    def test_dry_run_skips_webserver_spinup(self, monkeypatch: MonkeyPatch) -> None:
        self._mock_ray_return_value(monkeypatch, True)

        get_global_config_dict_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", get_global_config_dict_mock)

        ServerClient_mock = MagicMock(spec=ServerClient)
        monkeypatch.setattr(nemo_gym.server_utils, "ServerClient", ServerClient_mock)

        class TestSimpleServer(SimpleServer):
            def __init__(self, *args, **kwargs):
                pass

            def setup_webserver(self):
                assert False

            @classmethod
            def load_config_from_global_config(cls) -> None:
                pass

        TestSimpleServer.run_webserver()

    def test_setup_session_middleware_idempotent(self) -> None:
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from starlette.middleware.sessions import SessionMiddleware

        from nemo_gym.config_types import BaseRunServerInstanceConfig
        from nemo_gym.server_utils import SESSION_ID_KEY

        class TestSimpleServer(SimpleServer):
            def setup_webserver(self):
                assert False

        server = TestSimpleServer(
            config=BaseRunServerInstanceConfig(name="my_server", host="", port=0, entrypoint=""),
            server_client=ServerClient(
                head_server_config=BaseServerConfig(host="", port=0),
                global_config_dict=DictConfig({}),
            ),
        )

        app = FastAPI()
        server.setup_session_middleware(app)
        server.setup_session_middleware(app)

        session_middlewares = [m for m in app.user_middleware if m.cls is SessionMiddleware]
        assert 1 == len(session_middlewares)
        assert 2 == len(app.user_middleware)

        @app.get("/session")
        async def get_session(request: Request) -> dict:
            return {"session_id": request.session[SESSION_ID_KEY]}

        with TestClient(app) as client:
            response = client.get("/session")
            assert response.json()["session_id"]
            assert 1 == len(response.headers.get_list("set-cookie"))

    def _mock_global_client(self, monkeypatch: MonkeyPatch, connection_errors: int) -> MagicMock:
        """Global-client stand-in whose request() raises ClientOSError `connection_errors` times, then succeeds."""
        client = MagicMock()
        client.request = AsyncMock(side_effect=[ClientOSError()] * connection_errors + [client.success_response])
        monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", lambda: client)
        monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())
        return client

    async def test_request_bounded_connection_retries_surface_dead_endpoint(self, monkeypatch: MonkeyPatch) -> None:
        client = self._mock_global_client(monkeypatch, connection_errors=10)
        with raises(ClientOSError):
            await nemo_gym.server_utils.request("POST", "http://dead-host:1/v1", _max_connection_retries=3)
        assert client.request.await_count == 3

    async def test_request_connection_retries_unbounded_by_default(self, monkeypatch: MonkeyPatch) -> None:
        client = self._mock_global_client(monkeypatch, connection_errors=4)
        response = await nemo_gym.server_utils.request("POST", "http://flaky-host:1/v1")
        assert response is client.success_response
        assert client.request.await_count == 5

    def _make_map_client(self) -> ServerClient:
        return ServerClient(
            head_server_config=BaseServerConfig(host="head-host", port=12345),
            global_config_dict=DictConfig({"my_server": {"a": {"b": {"host": "old-host", "port": 1111}}}}),
        )

    def _reset_module_state(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.server_utils, "_HEAD_CONFIG_CACHE", {})
        monkeypatch.setattr(nemo_gym.server_utils, "_HEAD_CONFIG_FETCH_INFLIGHT", False)
        monkeypatch.setattr(nemo_gym.server_utils, "_HEAD_CONFIG_LAST_FAILURE", 0.0)

    @mark.parametrize(
        ("prime", "expect_host", "fetches"),
        [
            ("cache", "new-host", False),  # fresh shared-cache entry serves every client, no HTTP
            ("failure", "old-host", False),  # negative cache after a failed probe: keep the stale map, no stampede
            ("empty", "new-host", True),  # cold: one fetch from the head updates the map and the shared cache
        ],
    )
    async def test_refresh_global_config_paths(
        self, monkeypatch: MonkeyPatch, prime: str, expect_host: str, fetches: bool
    ) -> None:
        self._reset_module_state(monkeypatch)
        client = self._make_map_client()
        monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())
        if prime == "cache":
            cached_map = DictConfig({"my_server": {"a": {"b": {"host": "new-host", "port": 2222}}}})
            monkeypatch.setitem(
                nemo_gym.server_utils._HEAD_CONFIG_CACHE, ("head-host", 12345), (time.monotonic(), cached_map)
            )
        elif prime == "failure":
            monkeypatch.setattr(nemo_gym.server_utils, "_HEAD_CONFIG_LAST_FAILURE", time.monotonic())
        if fetches:
            fake_response = MagicMock()
            fake_response.content = b'{"my_server": {"a": {"b": {"host": "new-host", "port": 2222}}}}'

            async def fake_to_thread(fn, *args, **kwargs):
                return fake_response

            monkeypatch.setattr(nemo_gym.server_utils.asyncio, "to_thread", fake_to_thread)
        else:
            monkeypatch.setattr(
                nemo_gym.server_utils.requests, "get", MagicMock(side_effect=AssertionError("must not fetch"))
            )

        await client._refresh_global_config()

        assert client.global_config_dict["my_server"]["a"]["b"]["host"] == expect_host
        if fetches:
            assert ("head-host", 12345) in nemo_gym.server_utils._HEAD_CONFIG_CACHE

    async def test_request_reresolves_the_map_on_sustained_connection_errors(self, monkeypatch: MonkeyPatch) -> None:
        from aiohttp import ClientConnectionError

        self._reset_module_state(monkeypatch)
        client = self._make_map_client()
        seen_urls = []

        async def fake_request(method, url, **kwargs):
            seen_urls.append(url)
            if "old-host" in url:
                raise ClientConnectionError()
            response = MagicMock()
            response.status = 200
            return response

        monkeypatch.setattr(nemo_gym.server_utils, "request", fake_request)

        async def fake_refresh(self_client):
            self_client.global_config_dict = DictConfig(
                {"my_server": {"a": {"b": {"host": "new-host", "port": 2222}}}}
            )

        monkeypatch.setattr(ServerClient, "_refresh_global_config", fake_refresh)

        response = await client.post(server_name="my_server", url_path="/run")

        assert response.status == 200
        assert seen_urls == ["http://old-host:1111/run", "http://new-host:2222/run"]

    async def test_request_retries_a_stale_route_exactly_once(self, monkeypatch: MonkeyPatch) -> None:
        self._reset_module_state(monkeypatch)
        client = self._make_map_client()
        statuses = iter([404, 404])
        calls = {"n": 0}

        async def fake_request(method, url, **kwargs):
            calls["n"] += 1
            response = MagicMock()
            response.status = next(statuses)
            return response

        monkeypatch.setattr(nemo_gym.server_utils, "request", fake_request)
        refresh = AsyncMock()
        monkeypatch.setattr(ServerClient, "_refresh_global_config", refresh)

        response = await client.post(server_name="my_server", url_path="/run")

        # A second 404 is a real routing problem and is returned, not retried.
        assert response.status == 404
        assert calls["n"] == 2
        refresh.assert_awaited_once()

    @mark.parametrize("head_learns", [True, False])
    async def test_request_tolerates_convergence_but_bounds_a_missing_server(
        self, monkeypatch: MonkeyPatch, head_learns: bool
    ) -> None:
        """Transiently-missing server names re-resolve and recover; a name the
        head never publishes warns while retrying and finally raises the
        KeyError instead of spinning silently forever."""
        self._reset_module_state(monkeypatch)
        client = ServerClient(
            head_server_config=BaseServerConfig(host="head-host", port=12345),
            global_config_dict=DictConfig({}),
        )
        monkeypatch.setattr(nemo_gym.server_utils, "_SERVER_NAME_MISSING_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())

        async def fake_request(method, url, **kwargs):
            response = MagicMock()
            response.status = 200
            return response

        monkeypatch.setattr(nemo_gym.server_utils, "request", fake_request)
        refreshes = {"n": 0}

        async def fake_refresh(self_client):
            refreshes["n"] += 1
            if head_learns and refreshes["n"] == 2:
                self_client.global_config_dict = DictConfig(
                    {"my_server": {"a": {"b": {"host": "new-host", "port": 2222}}}}
                )

        monkeypatch.setattr(ServerClient, "_refresh_global_config", fake_refresh)

        if head_learns:
            response = await client.post(server_name="my_server", url_path="/run")
            assert response.status == 200
        else:
            with raises(KeyError):
                await client.post(server_name="my_server", url_path="/run")
            assert refreshes["n"] == 2  # bound 3: two re-resolve attempts, the third look-up raises
