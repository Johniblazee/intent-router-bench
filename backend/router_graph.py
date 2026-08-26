"""Minimal LangGraph chat router: classify intent -> route to workflow stub.

Usage: python router_graph.py [--model bge-small]
Swap --model for any key in bench.ROSTER to feel the latency difference live.
"""

import argparse
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from bench import DATASETS, ROSTER


class State(TypedDict):
    message: str
    intent: str
    response: str


def build_graph(clf, labels):
    def classify(state: State):
        return {"intent": clf.predict(state["message"])}

    def make_workflow(intent):
        # ponytail: stubs — replace each with the real workflow subgraph
        def workflow(state: State):
            return {"response": f"[{intent} workflow] handling: {state['message']!r}"}
        return workflow

    g = StateGraph(State)
    g.add_node("classify", classify)
    g.set_entry_point("classify")
    for intent in labels:
        g.add_node(intent, make_workflow(intent))
        g.add_edge(intent, END)
    g.add_conditional_edges("classify", lambda s: s["intent"], {i: i for i in labels})
    return g.compile()


DEMO_MESSAGES = [
    "hey, how's it going?",
    "my payment failed twice and I got charged anyway",
    "where's my package? it's been a week",
    "the app crashes every time I open settings",
    "I want to talk to a real human right now",
    "what's a good recipe for ramen?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bge-small", choices=list(ROSTER))
    args = ap.parse_args()

    train, _, labels = DATASETS["custom"]()
    clf = ROSTER[args.model]()
    print(f"loading {args.model} ...")
    clf.load()
    clf.train(train, labels)
    graph = build_graph(clf, labels)

    graph.invoke({"message": "warm up"})
    print(f"\nrouting with {args.model}:\n")
    for msg in DEMO_MESSAGES:
        t0 = time.perf_counter()
        out = graph.invoke({"message": msg})
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {ms:7.1f}ms  {out['intent']:<20} <- {msg!r}")
        print(f"            {out['response']}")


if __name__ == "__main__":
    main()
