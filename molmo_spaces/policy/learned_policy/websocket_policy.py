import logging
import time

import msgpack_numpy
import websockets.sync.client

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# The transport, as three functions.
#
# It lived inside WebsocketPolicy's methods, which meant the next consumer of
# "msgpack-numpy over a websocket to a model process" -- molmo_spaces/bridge --
# had to copy it. These are the same lines, named, so both callers share one
# implementation and one set of conventions:
#
#   * the server GREETS with a metadata frame the moment a client connects
#     (WebsocketPolicyServer already does this; bridge/client.py does too), so a
#     client learns who it is talking to before it sends anything;
#   * a STRING reply is the server's error channel, and is raised, never parsed;
#   * a refused connection is retried until connection_timeout, because the
#     usual reason is "the other terminal has not started yet"; any other
#     socket error is a real failure and is raised immediately.
# --------------------------------------------------------------------------- #
def ws_uri(host: str, port: int | None = None) -> str:
    """`host[:port]` -> a websocket uri, leaving an explicit scheme alone."""
    uri = host if host.startswith("ws") else f"ws://{host}"
    return f"{uri}:{port}" if port is not None else uri


def ws_connect(
    uri: str,
    *,
    connection_timeout: float | None = None,
    poll_s: float = 5.0,
    metadata_timeout: float = 10.0,
) -> tuple[websockets.sync.client.ClientConnection, dict]:
    """Connect, wait for the server's metadata frame, return both.

    Args:
        uri: full websocket uri (see ``ws_uri``).
        connection_timeout: give up after this many seconds of refusals;
            None waits forever, which is what you want when the peer is a
            terminal a human is about to start.
        poll_s: seconds between retries.
        metadata_timeout: seconds to wait for the greeting.
    """
    start = time.monotonic()
    try:
        while True:
            try:
                conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    open_timeout=600.0,
                    ping_interval=None,
                )
            except ConnectionRefusedError as e:
                if connection_timeout is not None and time.monotonic() - start > connection_timeout:
                    raise TimeoutError(f"Timeout waiting for server at {uri}") from e
                logger.info("Waiting for server at %s ...", uri)
                time.sleep(poll_s)
                continue
            return conn, msgpack_numpy.unpackb(conn.recv(timeout=metadata_timeout))
    except OSError as e:
        raise RuntimeError(f"Error waiting for server at {uri}: {e}") from e


def ws_request(
    conn: websockets.sync.client.ClientConnection, payload, *, timeout: float | None = None
):
    """One request, one reply. A string reply is the server's error channel."""
    conn.send(msgpack_numpy.packb(payload))
    response = conn.recv(timeout=timeout) if timeout is not None else conn.recv()
    if isinstance(response, str):
        # we're expecting bytes; if the server sends a string, it's an error.
        raise RuntimeError(f"Error in inference server:\n{response}")
    return msgpack_numpy.unpackb(response)


class WebsocketPolicy(InferencePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        config: MlSpacesExpConfig,
        model_name: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        connection_timeout: float | None = None,
    ):
        super().__init__(config)
        self.model_name = model_name
        self._last_prompt: str | None = None

        self._uri = ws_uri(host, port)
        self._ws = None
        self._server_metadata = None
        self._prepared = False
        self._connection_timeout = connection_timeout

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        return ws_connect(self._uri, connection_timeout=self._connection_timeout)

    def infer(self, obs: dict) -> dict:
        return ws_request(self._ws, obs, timeout=10)

    def reset(self) -> None:
        self.close()
        self._prepared = False
        self.prepare_model()

    def prepare_model(self) -> None:
        if not self._prepared:
            self._ws, self._server_metadata = self._wait_for_server()
            self._prepared = True

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            if len(obs) > 1:
                logger.warning(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "WARNING: obs list has %d elements but only using the first one!\n"
                    "This may indicate a batching issue - expected single observation.\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    len(obs),
                )
            obs = obs[0]
        model_input = {**obs}
        prompt = self.task.get_task_description()

        if self._last_prompt is None:
            self._last_prompt = prompt
        model_input["task"] = prompt
        return model_input

    def inference_model(self, model_input):
        self.prepare_model()
        return ws_request(self._ws, model_input)

    def model_output_to_action(self, model_output):
        action = {
            "arm": model_output["arm"],
            "gripper": model_output["gripper"],
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "websocket"
        info["policy_model_name"] = self.model_name
        info["prompt"] = self.task.get_task_description()
        return info

    def close(self):
        if self._ws is not None:
            logger.info("Closing websocket connection")
            self._ws.close()
            self._ws = None
