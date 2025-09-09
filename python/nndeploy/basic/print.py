# console_print_node.py
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
import typing as T

import nndeploy.dag
from nndeploy.base import Status


class ConsolePrintNode(nndeploy.dag.Node):
    """
    console node
    """
    def __init__(
        self,
        name: str,
        inputs: list[nndeploy.dag.Edge] = None,
        outputs: list[nndeploy.dag.Edge] = None
    ):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.ConsolePrintNode")
        self.set_desc("将输入内容打印到屏幕，可选日志/落盘")
        self.set_input_type(str)
        self.set_node_type(nndeploy.dag.NodeType.Output)
        self.set_io_type(nndeploy.dag.IOType.String)

        self.prefix: str = ""               # print prefix
        self.echo_to_stdout: bool = True    # stdout
        self.print_to_logger: bool = False  # logging
        self.logger_level: str = "INFO"     # DEBUG/INFO/WARNING/ERROR/CRITICAL
        self.save_to_file: bool = False     # save to disk
        self.file_path: str = ""            # path
        self.append_newline: bool = True    # newline

        self.frontend_params: dict = {
            "prefix": self.prefix,
            "echo_to_stdout": self.echo_to_stdout,
            "print_to_logger": self.print_to_logger,
            "logger_level": self.logger_level,
            "save_to_file": self.save_to_file,
            "file_path": self.file_path,
            "append_newline": self.append_newline,
        }

    def _sync_params_from_frontend(self):
        fp = self.frontend_params or {}
        self.prefix = str(fp.get("prefix", self.prefix))
        self.echo_to_stdout = bool(fp.get("echo_to_stdout", self.echo_to_stdout))
        self.print_to_logger = bool(fp.get("print_to_logger", self.print_to_logger))
        self.logger_level = str(fp.get("logger_level", self.logger_level)).upper()
        self.save_to_file = bool(fp.get("save_to_file", self.save_to_file))
        self.file_path = str(fp.get("file_path", self.file_path))
        self.append_newline = bool(fp.get("append_newline", self.append_newline))

    def run(self):
        in_edge = self.get_input(0)
        value = in_edge.get(self)

        self._sync_params_from_frontend()

        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)

        message = f"{self.prefix}{text}"

        if self.echo_to_stdout:
            print(message, flush=True)

        if self.print_to_logger:
            level = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }.get(self.logger_level, logging.INFO)
            logging.log(level, message)

        if self.save_to_file and self.file_path:
            try:
                path = Path(self.file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    if self.append_newline:
                        f.write(message + "\n")
                    else:
                        f.write(message)
            except Exception:
                logging.exception("[ConsolePrintNode] 写文件失败")

        return Status.ok()

    def serialize(self):
        base = json.loads(super().serialize())
        base["frontend_params"] = self.frontend_params
        return json.dumps(base, ensure_ascii=False)

    def deserialize(self, target: str):
        obj = json.loads(target)
        if "frontend_params" in obj:
            self.frontend_params = obj["frontend_params"] or self.frontend_params
        self._sync_params_from_frontend()
        return super().deserialize(target)


class ConsolePrintNodeCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        self.node: T.Optional[ConsolePrintNode] = None

    def create_node(
        self,
        name: str,
        inputs: list[nndeploy.dag.Edge],
        outputs: list[nndeploy.dag.Edge]
    ):
        self.node = ConsolePrintNode(name, inputs, outputs)
        return self.node


console_print_node_creator = ConsolePrintNodeCreator()
nndeploy.dag.register_node("nndeploy.basic.ConsolePrintNode", console_print_node_creator)
