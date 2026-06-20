from __future__ import annotations

import hashlib
import uuid
from pathlib import Path  # noqa: TC003 - runtime arg in identity builders


class InteractionIdentity:
    """
    Deterministic id derivation for ledger entities owned by one workflow.

    All ids are stable functions of the workflow id plus a small set of
    qualifier inputs. Centralized here so runner, graph nodes, recorder, and
    host-side code never duplicate format strings or digest implementations.

    IDs are opaque UUID strings. The scope and qualifiers are only used as
    UUID5 input material; they are never embedded in the stored identifier.
    """

    __DIGEST_LENGTH: int = 12

    @classmethod
    def stable(cls, *, scope: str, parts: tuple[object, ...]) -> str:
        """
        Return a deterministic opaque UUID for one logical identity.
        """

        material = "\x1f".join((scope, *(str(part) for part in parts)))
        return str(uuid.uuid5(uuid.NAMESPACE_URL, material))

    def __init__(self, *, workflow: str) -> None:
        """
        Bind the identity helper to one workflow id.
        """

        if not workflow:
            raise ValueError("InteractionIdentity requires a non-empty workflow id")

        self.__workflow = workflow

    @property
    def workflow(self) -> str:
        """
        Return the workflow id this identity helper is bound to.
        """

        return self.__workflow

    def task(self) -> str:
        """
        Return the root task id for the current workflow.
        """

        return self.stable(scope="task.root", parts=(self.__workflow,))

    def step_task(self, *, step_number: int, action_descriptor: str) -> str:
        """
        Return a deterministic sub-task id for one graph step attempt.
        """

        return self.stable(
            scope="task.step",
            parts=(self.__workflow, step_number, self.__digest(value=action_descriptor)),
        )

    def message(self, *, name: str) -> str:
        """
        Return a stable message id keyed only by name (e.g. ``request``).
        """

        return self.stable(scope="message", parts=(self.__workflow, name))

    def derived_message(self, *, name: str, qualifier: str) -> str:
        """
        Return a content-derived message id qualified by a hashed input.
        """

        return self.stable(
            scope="message.derived",
            parts=(self.__workflow, name, self.__digest(value=qualifier)),
        )

    def context(self, *, name: str) -> str:
        """
        Return a stable context recipe id keyed by name.
        """

        return self.stable(scope="context", parts=(self.__workflow, name))

    def membership(self, *, thread: str, role: str, actor: str) -> str:
        """
        Return a thread membership id for one actor in one role.
        """

        return self.stable(scope="membership", parts=(thread, role, actor))

    def artifact(self, *, path: Path) -> str:
        """
        Return a deterministic artifact id derived from the source path.
        """

        return self.stable(
            scope="artifact",
            parts=(self.__workflow, self.__digest(value=str(path))),
        )

    def script(self, *, path: Path) -> str:
        """
        Return a deterministic script id derived from the source path.
        """

        return self.stable(
            scope="script",
            parts=(self.__workflow, self.__digest(value=str(path))),
        )

    def job(self, *, name: str) -> str:
        """
        Return a stable background job id keyed by name.
        """

        return self.stable(scope="job", parts=(self.__workflow, name))

    def __digest(self, *, value: str) -> str:
        """
        Return a short, stable digest of the input string.
        """

        return hashlib.sha256(value.encode("utf-8")).hexdigest()[: self.__DIGEST_LENGTH]
