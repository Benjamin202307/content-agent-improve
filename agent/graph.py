from pathlib import Path
from dotenv import load_dotenv

# Always load the project's own .env, regardless of the directory used to
# launch uvicorn. This prevents optional nodes from silently losing config.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.graph import StateGraph, END
from agent.state import AgentState, Platform
from agent.nodes.pre_researcher import pre_researcher_node
from agent.nodes.planner import planner_node
from agent.nodes.researcher import researcher_node
from agent.nodes.writer import writer_node
from agent.nodes.critic import critic_node
from agent.nodes.paraphraser import paraphraser_node
from agent.nodes.image_fetcher import image_fetcher_node
from agent.nodes.screenshot_refiller import screenshot_refiller_node
from agent.memory import save as save_to_memory

# ─────────────────────────────────────────────────────────
# 构建 Graph
#
# 节点执行顺序：
#   pre_researcher → planner → researcher → writer → critic → 条件分支：
#     - score ≥ 7 或 retry ≥ 2 → paraphraser → image_fetcher → END
#     - score < 7 且 retry < 2  → 回到 researcher 重写
#
# ─────────────────────────────────────────────────────────


def should_retry(state: AgentState) -> str:
    """Critic 之后的条件分支：决定是否重写。"""
    score = state.get("critic_score", 7)
    retry_count = state.get("retry_count", 0)

    if score < 7 and retry_count < 2:
        print(f"\n⚠️  评分 {score}/10，不达标，准备第 {retry_count + 2} 次重写...")
        return "retry"
    else:
        if score >= 7:
            print(f"\n✅ 评分 {score}/10，质量达标，进入全文改写阶段")
        else:
            print(f"\n⚠️  评分 {score}/10，已达最大重试次数，跳过重写")
        return "pass"


def save_memory_node(state: AgentState) -> dict:
    """文章生成完毕后，把素材存入向量库供未来检索。"""
    print("\n[Memory] 保存素材到向量库...")
    try:
        save_to_memory(
            topic=state["topic"],
            context=state["context"],
            platform=state["platform"],
            topic_id=state.get("topic_id"),
        )
        message = "💾 素材已存入向量库"
    except Exception as exc:
        # Vector memory is optional and runs after the article, rewrite and
        # screenshots are already complete. An unavailable embedding model
        # must not discard the successfully generated article.
        print(f"  ⚠️ 向量素材保存失败，不影响文章交付：{type(exc).__name__}")
        message = "⚠️ 向量素材保存暂不可用，文章已正常生成，不受影响"
    return {"log": state.get("log", []) + [message]}


def increment_retry(state: AgentState) -> dict:
    """重试计数 +1，在回到 researcher 之前执行。"""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "log": state.get("log", []) + ["🔄 初稿不达标，重新搜索并重写..."],
    }


def should_retry_screenshots(state: AgentState) -> str:
    """截图阶段以实际成功数量为准，禁止交付不足五张的文章。"""
    if not state.get("needs_screenshot_retry"):
        return "pass"
    # Screenshot hosts may independently rate-limit or challenge a page. Give
    # the targeted refiller enough rounds to replace bad candidates while
    # keeping the article blocked until at least five real captures exist.
    if state.get("screenshot_retry_count", 0) < 4:
        return "retry"
    return "fail"


def screenshot_requirement_failed(state: AgentState) -> dict:
    count = state.get("screenshot_success_count", 0)
    return {
        "final_article": "",
        "log": state.get("log", []) + [
            f"截图要求未满足：实际仅成功 {count} 张。已停止交付，避免显示少于 5 张截图的文章。"
        ],
    }


workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("pre_researcher", pre_researcher_node)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("paraphraser", paraphraser_node)
workflow.add_node("increment_retry", increment_retry)
workflow.add_node("image_fetcher", image_fetcher_node)
workflow.add_node("screenshot_refiller", screenshot_refiller_node)
workflow.add_node("screenshot_requirement_failed", screenshot_requirement_failed)
workflow.add_node("save_memory", save_memory_node)

# 连接边
workflow.set_entry_point("pre_researcher")
workflow.add_edge("pre_researcher", "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "critic")

# 条件分支：Critic 之后
workflow.add_conditional_edges(
    "critic",
    should_retry,
    {
        "retry": "increment_retry",
        "pass": "paraphraser",
    },
)
workflow.add_edge("increment_retry", "researcher")
workflow.add_edge("paraphraser", "image_fetcher")
workflow.add_conditional_edges(
    "image_fetcher",
    should_retry_screenshots,
    {
        "retry": "screenshot_refiller",
        "pass": "save_memory",
        "fail": "screenshot_requirement_failed",
    },
)
workflow.add_edge("screenshot_refiller", "image_fetcher")
workflow.add_edge("save_memory", END)
workflow.add_edge("screenshot_requirement_failed", END)

graph = workflow.compile()


def _initial_state() -> dict:
    """每次调用返回全新的初始状态，避免可变默认值被复用。"""
    return {
        "outline": "",
        "keywords": [],
        "raw_materials": [],
        "context": "",
        "draft": "",
        "paraphrase_applied": False,
        "paraphrase_request_count": 0,
        "paraphrase_changed_count": 0,
        "paraphrase_request_ids": [],
        "paraphrase_error": "",
        "images": {},
        "final_article": "",
        "log": [],
        "critic_score": 0,
        "history_context": "",
        "critic_feedback": "",
        "retry_count": 0,
        "screenshot_retry_count": 0,
        "screenshot_retry_note": "",
        "screenshot_success_count": 0,
        "needs_screenshot_retry": False,
        "screenshot_source_urls": [],
        "screenshot_attempted_urls": [],
    }


def run(topic: str, platform: Platform, direction: str = "tech") -> dict:
    """
    对外暴露的统一入口（阻塞式）。
    返回 { "article": str, "log": list[str], "score": int }
    """
    result = graph.invoke({
        "topic": topic,
        "platform": platform,
        "direction": direction,
        **_initial_state(),
    })

    return {
        "article": result["final_article"],
        "log": result["log"],
        "score": result["critic_score"],
    }


_NEXT_NODE = {
    "pre_researcher": "planner",
    "planner": "researcher",
    "researcher": "writer",
    "writer": "critic",
    "paraphraser": "image_fetcher",
    "image_fetcher": "save_memory",
    "screenshot_refiller": "image_fetcher",
    "screenshot_requirement_failed": "",
    "save_memory": "",
}


def run_stream(topic: str, platform: Platform, direction: str = "tech", image_style: str | None = None, topic_id: int | None = None):
    """
    流式入口，yield 每个节点的输出。
    每次 yield 一个 dict: { "node": str, "data": dict, "active": str }
    active 表示当前正在执行的节点（即下一个节点），用于前端状态同步。
    """
    init = {"topic": topic, "platform": platform, "direction": direction, **_initial_state()}
    if image_style:
        init["image_style"] = image_style
    if topic_id is not None:
        init["topic_id"] = topic_id

    retry_count = 0
    for event in graph.stream(
        init,
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            if node_name == "increment_retry":
                retry_count += 1

            # 计算当前正在运行的节点（下一个节点）
            if node_name == "critic":
                score = node_output.get("critic_score", 7)
                if score < 7 and retry_count < 2:
                    active = "researcher"
                else:
                    active = "paraphraser"
            elif node_name == "image_fetcher":
                if node_output.get("needs_screenshot_retry"):
                    active = "screenshot_refiller"
                else:
                    active = "save_memory"
            else:
                active = _NEXT_NODE.get(node_name, "")

            yield {"node": node_name, "data": node_output, "active": active}
