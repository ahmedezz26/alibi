from alibi.routing import trace_kind
from alibi.types import Step


def step(kind, text, name=None):
    return Step("0", kind, None, {"content": text}, name=name)


def test_agent_editing_files_and_running_commands_is_code():
    steps = [
        step("user", "Fix the failing test in the repo."),
        step("assistant", "```bash\ngrep -rn parse src/\n```"),
        step("tool", "src/parse.py:10: def parse(x):"),
        step("assistant", "```str_replace path=/app/src/parse.py\nold\nnew\n```"),
        step("tool", "edited"),
        step("assistant", 'Tool calls:\nexecute_bash({\n  "command": "pytest -q"\n})'),
    ]
    assert trace_kind(steps) == "code"


def test_agent_talking_with_a_customer_and_calling_business_tools_is_conversation():
    steps = [
        step("system", "# Airline Agent Policy. You can help users book flights."),
        step("assistant", "Hi! How can I help you today?"),
        step("user", "My flight was delayed and I want compensation."),
        step("assistant", "I'm sorry to hear that. Could you share your user ID?"),
        step("user", "Sure, it's mei_brown_7075."),
        step("assistant", '[tool_calls] - get_user_details({"user_id": "mei_brown_7075"})'),
        step("tool", '{"name": "Mei Brown", "membership": "gold"}'),
    ]
    assert trace_kind(steps) == "conversation"


def test_customer_messages_relayed_by_the_terminal_still_count_as_conversation():
    steps = [
        step("assistant", "你好，请问需要什么服务？", "Action_Expert"),
        step("tool", "你好，我的面已经送到公司了，能处理一下吗？", "Computer_terminal"),
        step("assistant", "Tool call: get_user_all_orders Arguments: {}", "Action_Expert"),
        step(
            "tool", "Tool result from get_user_all_orders: Order(order_id:1)", "Computer_terminal"
        ),
        step("assistant", "好的，我已经帮您查到订单了，请问要取消吗？", "Action_Expert"),
        step("tool", "是的，请取消。", "Computer_terminal"),
    ]
    assert trace_kind(steps) == "conversation"


def test_agents_planning_among_themselves_is_other():
    steps = [
        step("tool", "Plan this travel sample with the available tools."),
        step("assistant", "Please find flights from Chicago to Ogdensburg.", "Task_Planner"),
        step("assistant", 'Tool call: FlightSearch Arguments: {"from": "Chicago"}', "agent"),
        step("tool", "Tool result from FlightSearch: F123 $200"),
        step("assistant", "Flight F123 costs $200.", "transport_agent"),
    ]
    assert trace_kind(steps) == "other"
