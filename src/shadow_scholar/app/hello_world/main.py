from shadow_scholar.cli import cli, safe_import, Argument


with safe_import():
    import gradio as gr


@cli(
    'app.hello_world',
    arguments=[
        Argument('--host', default='0.0.0.0', help='Host to bind to'),
        Argument('--port', default=7860, help='Port to bind to')
    ],
    requirements=['gradio'],
)
def run_hello_world(host: str, port: int):
    with gr.Blocks() as app:
        gr.Markdown(
            "# 🕶️ Shadow Scholar 🎓\n"
            "Hello, world!"
        )

    try:
        app.launch(server_name=host, server_port=port)
    except Exception as e:
        gr.close_all()
        raise e
