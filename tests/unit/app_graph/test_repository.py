import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.app_graph.models import Screen, Transition
from fathom.app_graph.state import Completion, Identity, Observation, Start
from fathom.infrastructure.apps.repository import Repository


class TestRepository(unittest.IsolatedAsyncioTestCase):
    async def test_two_blocks_survive_restart_and_replay(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Repository(root=Path(directory))
            identity = Identity(tenant="org", package="com.example")
            first = Observation(
                identity=identity,
                execution="home-block",
                sequence=0,
                goal="Explore Home",
                screen=Screen(hash="home", name="Home", activity="Home"),
                image=b"home",
                xml='<hierarchy><node package="com.example" resource-id="com.example:id/search" text="Search" clickable="true"/></hierarchy>',
            )
            await repository.start(
                start=Start(identity=identity, execution="home-block", goal="Explore Home")
            )
            await repository.observe(observation=first)
            await repository.observe(observation=first)
            await repository.complete(
                completion=Completion(identity=identity, execution="home-block")
            )
            await repository.observe(
                observation=Observation(
                    identity=identity,
                    execution="cart-block",
                    sequence=1,
                    goal="Explore Cart",
                    screen=Screen(hash="cart", name="Cart", activity="Cart"),
                    image=b"cart",
                    transition=Transition(
                        source_hash="home",
                        target_hash="cart",
                        action_type="tap",
                        action_target="Cart",
                    ),
                )
            )
            snapshot = await Repository(root=Path(directory)).read(identity=identity)
            self.assertEqual(len(snapshot.screens), 2)
            self.assertEqual(len(snapshot.transitions), 1)
            self.assertEqual(len(snapshot.blocks), 2)
            self.assertEqual(snapshot.exploration_count, 2)
            self.assertEqual(snapshot.screens[0].visits, 1)
            self.assertEqual(
                snapshot.screens[0].elements[0].locators[0].value, "com.example:id/search"
            )
            self.assertEqual(Path(snapshot.screens[0].screenshot_path).read_bytes(), b"home")
            self.assertEqual(snapshot.blocks[0].status, "completed")

    async def test_concurrent_blocks_do_not_overwrite_each_other(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Repository(root=Path(directory))
            identity = Identity(tenant="org", package="com.example")
            await asyncio.gather(
                *(
                    repository.observe(
                        observation=Observation(
                            identity=identity,
                            execution=f"block-{index}",
                            sequence=0,
                            goal="Explore",
                            image=b"capture",
                            screen=Screen(hash=str(index), name=str(index), activity=str(index)),
                        )
                    )
                    for index in range(8)
                )
            )
            snapshot = await repository.read(identity=identity)
            self.assertEqual(len(snapshot.screens), 8)
            self.assertEqual(len(snapshot.blocks), 8)

    async def test_tenants_are_isolated_and_failed_block_keeps_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Repository(root=Path(directory))
            identity = Identity(tenant="one", package="com.example")
            await repository.observe(
                observation=Observation(
                    identity=identity,
                    execution="block",
                    sequence=0,
                    goal="Explore",
                    image=b"capture",
                    screen=Screen(hash="a", name="Home", activity="Home"),
                )
            )
            await repository.complete(
                completion=Completion(identity=identity, execution="block", error="Cancelled")
            )
            self.assertEqual(await repository.catalog(tenant="two"), [])
            other = await repository.read(identity=Identity(tenant="two", package="com.example"))
            self.assertEqual(other.screens, [])
            own = await repository.read(identity=identity)
            self.assertEqual(own.blocks[0].status, "failed")
            self.assertEqual(len(own.screens), 1)
