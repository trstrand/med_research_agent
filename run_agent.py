
import os
import google.auth
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from med_research_app.agent import root_agent

def run_agent():
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # Ask the researcher specifically
    message = types.Content(
        role="user", parts=[types.Part.from_text(text="Research PubMed for the latest on valerian root benefits and summarize it.")]
    )

    print("Running agent...")
    try:
        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
            )
        )
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent: {part.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_agent()
