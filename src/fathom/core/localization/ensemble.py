from __future__ import annotations

import asyncio
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants.perception import (
    ENSEMBLE_IOU_AGREEMENT_FLOOR,
    ENSEMBLE_MIN_AGREEING_MEMBERS,
    ENSEMBLE_SINGLE_PROPOSAL_CONFIDENCE_FLOOR,
)
from fathom.interfaces.localization import TargetLocalizerPort
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationProposal
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


@dataclass(frozen=True)
class _MemberOutcome:
    """
    Result of invoking one localization member.
    """

    proposal: Optional[LocalizationProposal]
    failed: bool = False


@dataclass(frozen=True)
class _ProposalCollection:
    """
    Aggregated localizer proposals and member health.
    """

    proposals: List[LocalizationProposal]
    failed_members: int = 0


class EnsembleLocalizerService:
    """
    Aggregates concurrent target-localizer proposals by IoU-clustering consensus.
    """

    def __init__(
        self,
        *,
        members: Tuple[TargetLocalizerPort, ...] = (),
        agreement_floor: float = ENSEMBLE_IOU_AGREEMENT_FLOOR,
        minimum_agreeing: int = ENSEMBLE_MIN_AGREEING_MEMBERS,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the ensemble with members, quorum thresholds, and run context.
        """

        self.__members = members
        self.__agreement_floor = agreement_floor
        self.__minimum_agreeing = minimum_agreeing
        self.__workflow_id = workflow_id

    @property
    def members(self) -> Tuple[TargetLocalizerPort, ...]:
        """
        Return the configured ensemble members.
        """

        return self.__members

    async def locate(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
        capture: ScreenCapture,
        budget: LocalizationBudget,
    ) -> Optional[LocalizationProposal]:
        """
        Run every enabled member concurrently and return the consensus proposal.
        """

        if not self.__members:
            return None

        context = self.__log_context(
            activity=capture.activity,
            target=self.__target_text(action=action),
        )
        logger.info(
            "Ensemble locate started",
            extra={
                **context,
                "event": "ensemble.locate.started",
                "member.count": len(self.__members),
            },
        )

        collection = await self.__collect_proposals(
            action=action,
            observation=observation,
            capture=capture,
            budget=budget,
        )
        proposals = collection.proposals
        if len(proposals) < self.__minimum_agreeing:
            if single := self.__single_confident_proposal(
                proposals=proposals,
                failed_members=collection.failed_members,
                budget=budget,
            ):
                logger.info(
                    "Ensemble accepted single high-confidence proposal",
                    extra={
                        **context,
                        "event": "ensemble.locate.single_confident",
                        "proposal.source": single.source,
                        "proposal.confidence": single.confidence,
                    },
                )
                return single

            self.__log_disagreement(
                context=context,
                proposals=proposals,
                reason="insufficient.proposals",
            )
            return None

        if (cluster := self.__strongest_cluster(proposals=proposals)) is None:
            self.__log_disagreement(
                context=context,
                proposals=proposals,
                reason="no.cluster.found",
            )
            return None

        if len(cluster) < self.__minimum_agreeing:
            self.__log_disagreement(
                context=context,
                proposals=proposals,
                reason="no.consensus.cluster",
            )
            return None

        consensus = self.__fuse(cluster=cluster, total=len(self.__members))
        logger.info(
            "Ensemble consensus reached",
            extra={
                **context,
                "event": "ensemble.locate.completed",
                "cluster.size": len(cluster),
                "members.agreeing": [proposal.source for proposal in cluster],
                "consensus.confidence": consensus.confidence,
            },
        )
        return consensus

    async def __collect_proposals(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
        capture: ScreenCapture,
        budget: LocalizationBudget,
    ) -> _ProposalCollection:
        """
        Invoke every member concurrently and retain member failure state.
        """

        coroutines = [
            self.__invoke_member(
                member=member,
                action=action,
                observation=observation,
                capture=capture,
                budget=budget,
            )
            for member in self.__members
        ]
        outcomes = await asyncio.gather(*coroutines, return_exceptions=False)
        return _ProposalCollection(
            proposals=[outcome.proposal for outcome in outcomes if outcome.proposal is not None],
            failed_members=sum(1 for outcome in outcomes if outcome.failed),
        )

    @staticmethod
    def __single_confident_proposal(
        *,
        proposals: List[LocalizationProposal],
        failed_members: int,
        budget: LocalizationBudget,
    ) -> Optional[LocalizationProposal]:
        """
        Return a lone proposal only when it is above the high-confidence floor.
        """

        if len(proposals) != 1 or failed_members > 0:
            return None

        proposal = proposals[0]
        floor = max(budget.threshold, ENSEMBLE_SINGLE_PROPOSAL_CONFIDENCE_FLOOR)
        if proposal.confidence < floor:
            return None

        return proposal

    async def __invoke_member(
        self,
        *,
        member: TargetLocalizerPort,
        action: Action,
        observation: ScreenObservation,
        capture: ScreenCapture,
        budget: LocalizationBudget,
    ) -> _MemberOutcome:
        """
        Call one member with its own wait_for window, isolating its failures.
        """

        timeout = max(0.001, budget.local / 1000.0)
        context = self.__log_context(activity=capture.activity, target="")
        try:
            proposal = await asyncio.wait_for(
                member.locate(
                    action=action,
                    budget=budget,
                    capture=capture,
                    observation=observation,
                ),
                timeout=timeout,
            )
            return _MemberOutcome(proposal=proposal)
        except asyncio.TimeoutError:
            logger.warning(
                "Ensemble member timed out",
                extra={
                    **context,
                    "member": member.name,
                    "timeout.seconds": timeout,
                    "event": "ensemble.member.timeout",
                },
            )
            return _MemberOutcome(proposal=None, failed=True)
        except Exception:
            # Localization is an optional, fail-soft enrichment;
            # log the full traceback and let other members vote.
            logger.exception(
                "Ensemble member raised — degrading to no proposal",
                extra={
                    **context,
                    "member": member.name,
                    "event": "ensemble.member.error",
                },
            )
            return _MemberOutcome(proposal=None, failed=True)

    def __strongest_cluster(
        self,
        *,
        proposals: List[LocalizationProposal],
    ) -> Optional[List[LocalizationProposal]]:
        """
        Return the largest IoU-coherent cluster of proposals.
        """

        clusters: List[List[LocalizationProposal]] = []
        for proposal in proposals:
            if (target := self.__find_cluster(proposal=proposal, clusters=clusters)) is None:
                clusters.append([proposal])
            else:
                target.append(proposal)

        if not clusters:
            return None

        return max(clusters, key=len)

    def __find_cluster(
        self,
        *,
        proposal: LocalizationProposal,
        clusters: List[List[LocalizationProposal]],
    ) -> Optional[List[LocalizationProposal]]:
        """
        Return the first cluster whose representative IoU with the proposal clears the floor.
        """

        for cluster in clusters:
            representative = cluster[0]
            if (
                self.__iou(first=proposal.bounds, second=representative.bounds)
                >= self.__agreement_floor
            ):
                return cluster
        return None

    def __fuse(
        self,
        *,
        cluster: List[LocalizationProposal],
        total: int,
    ) -> LocalizationProposal:
        """
        Build the consensus proposal from a coherent cluster.
        """

        bounds = self.__mean_bounds(cluster=cluster)
        confidence = self.__cluster_confidence(cluster=cluster, total=total)
        source = ",".join(sorted({proposal.source for proposal in cluster}))
        return LocalizationProposal(
            bounds=bounds,
            confidence=confidence,
            source=source,
            rationale=(
                f"Ensemble consensus across {len(cluster)}/{total} members "
                f"(agreement floor {self.__agreement_floor})."
            ),
        )

    @staticmethod
    def __mean_bounds(*, cluster: List[LocalizationProposal]) -> Bounds:
        """
        Compute the arithmetic-mean pixel bounds of a coherent cluster.
        """

        count = len(cluster)
        x_total = sum(proposal.bounds.x for proposal in cluster)
        y_total = sum(proposal.bounds.y for proposal in cluster)
        width_total = sum(proposal.bounds.width for proposal in cluster)
        height_total = sum(proposal.bounds.height for proposal in cluster)
        return Bounds(
            x=x_total // count,
            y=y_total // count,
            source=CoordinateSource.MODEL,
            width=max(1, width_total // count),
            height=max(1, height_total // count),
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    @staticmethod
    def __cluster_confidence(*, cluster: List[LocalizationProposal], total: int) -> float:
        """
        Scale the cluster mean confidence by its share of the total ensemble.
        """

        mean = sum(proposal.confidence for proposal in cluster) / len(cluster)
        coverage = len(cluster) / max(1, total)

        return float(min(1.0, mean * coverage))

    @staticmethod
    def __iou(*, first: Bounds, second: Bounds) -> float:
        """
        Return the intersection-over-union for two pixel bounds.
        """

        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        first_area = first.width * first.height
        second_area = second.width * second.height
        union = first_area + second_area - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)

    def __log_context(self, *, activity: str, target: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for ensemble entries.
        """

        return {
            "component": "core.localization.ensemble",
            "workflow.id": self.__workflow_id,
            "activity": activity,
            "target": target[:80],
        }

    def __log_disagreement(
        self,
        *,
        context: Dict[str, Any],
        proposals: List[LocalizationProposal],
        reason: str,
    ) -> None:
        """
        Emit a structured log entry summarizing why the ensemble produced no consensus.
        """

        logger.info(
            "Ensemble disagreement",
            extra={
                **context,
                "event": "ensemble.disagreement",
                "reason": reason,
                "proposal.count": len(proposals),
                "members.proposing": [proposal.source for proposal in proposals],
            },
        )

    @staticmethod
    def __target_text(*, action: Action) -> str:
        """
        Return the semantic target text used for log correlation.
        """

        return (
            action.natural_language_target
            or action.export_target
            or action.script_target
            or action.target
            or ""
        ).strip()
