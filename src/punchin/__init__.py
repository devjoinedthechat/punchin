"""Fork a recorded voice-agent call at the turn it went wrong.

The pieces, in the order a call goes through them:

    Scenario   a customer with a goal, and the outcome that is right for them
    record     run an agent against one, optionally spoken down a phone line
    Call       what was said, what was heard, what was called, what was booked
    extract    read the customer's goal state back out of a recording
    fork       re-run the recording from one turn, with the change applied
    check      fail when a run is worse than the baseline

`CommandAgent` is the way in for an agent that is not punchin's own.
"""

from punchin.adapter import PROTOCOL, AgentProtocolError, CommandAgent
from punchin.agent import Agent, AgentTurn, Lead, ModelAgent, ScriptedAgent
from punchin.call import FORMAT, Call, ToolCall, Turn
from punchin.check import Regression, Rule, baseline_from, compare
from punchin.customer import Customer, CustomerTurn, ScriptedCustomer
from punchin.dms import Dms, Vehicle, fresh
from punchin.fidelity import teacher_forced
from punchin.fork import Budget, ForkReport, fork, fork_once
from punchin.goal import extract
from punchin.metrics import feel, outcome, summarize
from punchin.model import ClaudeCodeModel, Completion, Model, ModelDidNotRun
from punchin.pinned import PinnedCustomer
from punchin.record import converse, record
from punchin.scenarios import SCENARIOS, GoalState, Scenario

__version__ = "0.1.0.dev0"

__all__ = [
    "FORMAT",
    "PROTOCOL",
    "SCENARIOS",
    "Agent",
    "AgentProtocolError",
    "AgentTurn",
    "Budget",
    "Call",
    "ClaudeCodeModel",
    "CommandAgent",
    "Completion",
    "Customer",
    "CustomerTurn",
    "Dms",
    "ForkReport",
    "GoalState",
    "Lead",
    "Model",
    "ModelAgent",
    "ModelDidNotRun",
    "PinnedCustomer",
    "Regression",
    "Rule",
    "Scenario",
    "ScriptedAgent",
    "ScriptedCustomer",
    "ToolCall",
    "Turn",
    "Vehicle",
    "__version__",
    "baseline_from",
    "compare",
    "converse",
    "extract",
    "feel",
    "fork",
    "fork_once",
    "fresh",
    "outcome",
    "record",
    "summarize",
    "teacher_forced",
]
