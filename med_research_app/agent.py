# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import LlmAgent
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.integrations.bigquery.config import BigQueryToolConfig
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, SseConnectionParams, StreamableHTTPConnectionParams
from google.adk.models import Gemini
from google.genai import types
from mcp import StdioServerParameters
from pydantic import BaseModel
from google.adk.apps import App, ResumabilityConfig
import os
import google.auth

# Auth Setup
auth_credentials, auth_project_id = google.auth.default()
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or auth_project_id
os.environ["GOOGLE_CLOUD_PROJECT"] = str(project_id)
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# 0. Configure Shared Model with Retry Logic
shared_model = Gemini(
    model="gemini-2.5-flash",
    retry_options=types.HttpRetryOptions(attempts=5),
)

# 1. Define Communication Schemas
class ResearchOutput(BaseModel):
    findings: str
    sources: list[str]
    sql_queries: list[str]

# 2. Configure Toolsets
bigquery_toolset = BigQueryToolset(
    bigquery_tool_config=BigQueryToolConfig(compute_project_id=project_id)
)

maps_mcp = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mapstools.googleapis.com/mcp",
        headers={
            "x-goog-api-key": os.environ.get("GOOGLE_MAPS_API_KEY", "MISSING_KEY"),
        }
    )
)

workspace_mcp = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://calendarmcp.googleapis.com/mcp/v1",
    )
)

# 3. Define Specialized Agents

summarizer = LlmAgent(
    name="summarizer",
    model=shared_model,
    description="Summarizes complex medical research findings into clear clinical reports.",
    instruction="""You are a senior medical science liaison.
    Your goal is to take research findings and produce a clear, professional, and actionable summary for clinicians.
    Maintain high medical accuracy and cite sources provided.
    When done, let the user know you've finished the summary and are ready for further questions or to return to the coordinator.""",
)

researcher = LlmAgent(
    name="researcher",
    model=shared_model,
    description="Researches PubMed PMC via BigQuery and provides summarized findings.",
    instruction="""You are an expert medical researcher.
    Your goal is to query the PubMed PMC public dataset in BigQuery (`bigquery-public-data.pmc_open_access_commercial`) to find relevant information.
    
    The dataset contains two main tables:
    1. `pmc_metadata`: Metadata for articles.
    2. `articles`: The full text or content of the articles.
    
    IMPORTANT: 
    1. Always limit SQL queries to LIMIT 5.
    2. AFTER you get findings, you MUST call the 'summarizer' to process them before returning to the user.
    3. Do NOT provide raw data or JSON to the user directly; always use the summarizer.""",
    tools=[bigquery_toolset],
    sub_agents=[summarizer]
)

maps_agent = LlmAgent(
    name="maps_agent",
    model=shared_model,
    description="Geospatial expert for finding locations, medical facilities, or stores.",
    instruction="""You are a geospatial expert. 
    Use Google Maps tools to find locations, businesses, and provide routing or distance information.
    Help the user find where they can purchase supplements or find medical services.""",
    tools=[maps_mcp],
)

workspace_agent = LlmAgent(
    name="workspace_agent",
    model=shared_model,
    description="Productivity assistant for Google Calendar.",
    instruction="""You are a productivity assistant with access to Google Calendar.
    You can list events, create meetings, and check availability.
    Help the user coordinate meetings related to their research.""",
    tools=[workspace_mcp],
)

orchestrator = LlmAgent(
    name="orchestrator",
    model=shared_model,
    instruction="""You are a highly capable and versatile personal medical research assistant. 
    Your goal is to be the central hub for the user's research journey, from gathering scientific data to practical logistics and organization.
    
    Be proactive and collaborative:
    - If the user asks about the benefits or science of something (e.g., 'valerian root'), transfer to the 'researcher'.
    - If the user wants to know where to buy something, find local stores, or get directions, transfer to the 'maps_agent'.
    - If the user needs to organize their findings, create calendar events, or check availability, transfer to the 'workspace_agent'.
    
    Never tell the user you 'only' do one thing. If they have a request that falls outside these areas, try to find a way to help or explain how one of your specialized agents could assist.
    
    Always greet the user warmly and offer to help them with both their scientific research and their practical next steps.""",
    sub_agents=[researcher, maps_agent, workspace_agent]
)

# 4. Wrap in an App
app = App(
    name="med_research_app",
    root_agent=orchestrator,
    resumability_config=ResumabilityConfig(is_resumable=True)
)

# For compatibility with tools that look for root_agent
root_agent = orchestrator
