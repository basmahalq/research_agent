import uuid

import gradio as gr

from agent import build_graph, empty_state, get_checkpointer

memory, _checkpoint_cm = get_checkpointer()
agent = build_graph().compile(checkpointer=memory)


def respond(message, thread_id, log):
    if not message.strip():
        return log, thread_id, ""

    if thread_id is None:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke(empty_state(message), config=config)
    else:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.get_state(config)
        result = agent.invoke(
            {**state.values, "query": message, "iterations": 0, "evaluation": ""},
            config=config,
        )

    entry = f"**{message}**\n\n{result['report']}\n\n---\n"
    new_log = (log + "\n" + entry) if log else entry
    return new_log, thread_id, ""


def new_session():
    return "", None, ""


with gr.Blocks(title="Research Agent") as demo:
    gr.Markdown("## Research Agent")
    gr.Markdown("Enter a topic to research. Ask follow-ups the same way, right below.")

    thread_state = gr.State(None)
    output = gr.Markdown()

    with gr.Row():
        msg = gr.Textbox(placeholder="Enter a topic, or ask a follow-up", scale=5, show_label=False)
        send_btn = gr.Button("Send", scale=1)

    clear_btn = gr.Button("New research")

    send_btn.click(respond, [msg, thread_state, output], [output, thread_state, msg])
    msg.submit(respond, [msg, thread_state, output], [output, thread_state, msg])
    clear_btn.click(new_session, None, [output, thread_state, msg])


if __name__ == "__main__":
    demo.launch()