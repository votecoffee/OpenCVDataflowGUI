"""Stable dependency planning and bounded execution of ready branches."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from heapq import heappop, heappush
from os import cpu_count

MAX_BRANCH_WORKERS = 8


def automatic_workers():
    return min(4, cpu_count() or 1)


class DependencyPlan:
    def __init__(self, keys, connections):
        positions = {key: index for index, key in enumerate(keys)}
        self.incoming = {key: [] for key in positions}
        self.parents = {key: set() for key in positions}
        self.children = {key: [] for key in positions}
        for connection in connections:
            source, target = connection.source, connection.target
            self.incoming[target].append(connection)
            if source not in self.parents[target]:
                self.parents[target].add(source)
                self.children[source].append(target)

        counts = {key: len(parents) for key, parents in self.parents.items()}
        ready = [key for key, count in counts.items() if count == 0]
        self.order = []
        self.can_overlap = False
        while ready:
            self.can_overlap |= len(ready) > 1
            self.order.extend(ready)
            following = []
            for key in ready:
                for child in self.children[key]:
                    counts[child] -= 1
                    if counts[child] == 0:
                        following.append(child)
            ready = sorted(following, key=positions.__getitem__)
        if len(self.order) != len(positions):
            raise ValueError("Cycles are not supported")

    def run(self, execute, publish, provided, eligible, workers, cancelled, skip):
        remaining = [key for key in self.order if key not in provided]
        if workers == 1 or not eligible or not self.can_overlap:
            for key in remaining:
                publish(key, execute(key))
            return

        # Prefer work that unlocks longer downstream chains. Break ties by graph order.
        length = {}
        for key in reversed(self.order):
            length[key] = 1 + max((length[child] for child in self.children[key]), default=0)
        position = {key: index for index, key in enumerate(self.order)}
        counts = {key: len(self.parents[key] - provided.keys()) for key in remaining}
        owners = deque(key for key in remaining if key not in eligible)
        ready = []
        running = {}

        def enqueue(key):
            if counts[key] == 0 and key in eligible:
                heappush(ready, (-length[key], position[key], key))

        def complete(key, output):
            del counts[key]
            publish(key, output)
            for child in self.children[key]:
                if child in counts:
                    counts[child] -= 1
                    enqueue(child)

        for key in remaining:
            enqueue(key)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="opencv-branch") as pool:
            while ready or owners or running:
                if cancelled():
                    for key in counts.keys() - set(running.values()):
                        skip(key)
                    for future in running:
                        future.result()
                    return

                for future in tuple(running):
                    if future.done():
                        complete(running.pop(future), future.result())
                while ready and len(running) < workers and not cancelled():
                    _, _, key = heappop(ready)
                    running[pool.submit(execute, key)] = key
                if owners and counts[owners[0]] == 0 and not cancelled():
                    key = owners.popleft()
                    complete(key, execute(key))
                elif running:
                    wait(running, return_when=FIRST_COMPLETED)
