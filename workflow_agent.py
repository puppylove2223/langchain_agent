import asyncio
import base64
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated

import pyautogui
from PIL import Image
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class WorkflowStep(BaseModel):
    step_number: int
    action: str
    motivation: str
    ui_elements: List[str]
    timestamp: str
    screenshot_path: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class WorkflowState(TypedDict):
    session_id: str
    screenshots: List[str]
    steps: List[WorkflowStep]
    current_step: int
    analysis_complete: bool
    needs_human_input: bool
    human_question: str
    continue_workflow: bool

class WorkflowAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1000
        )
        self.session_folder = Path("sessions")
        self.session_folder.mkdir(exist_ok=True)
    
    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks"""
        response_text = response_content.strip()
        if response_text.startswith("```json"):
            # Remove markdown formatting
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            # Handle generic code blocks
            response_text = response_text.replace("```", "").strip()
        return response_text
        
    def create_session(self) -> str:
        """Create a new session with unique ID"""
        session_id = str(uuid.uuid4())[:8]
        session_path = self.session_folder / session_id
        session_path.mkdir(exist_ok=True)
        (session_path / "screenshots").mkdir(exist_ok=True)
        return session_id

    def capture_screenshot(self, session_id: str, step_number: int) -> str:
        """Capture screenshot and save to session folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{step_number}_{timestamp}.png"
        filepath = self.session_folder / session_id / "screenshots" / filename
        
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        
        print(f"📸 Screenshot captured: {filename}")
        return str(filepath)

    def analyze_screenshot(self, screenshot_path: str, previous_steps: List[WorkflowStep]) -> Dict:
        """Analyze screenshot to determine UI changes and motivations using pure LLM reasoning"""
        
        # Create rich context from previous steps
        context = self._build_workflow_context(previous_steps)
        
        # First pass: Basic analysis
        analysis_prompt = f"""
        You are analyzing a screenshot as part of an ongoing workflow documentation process.
        
        WORKFLOW CONTEXT:
        {context}
        
        TASK: Analyze this screenshot to understand what happened and why.
        
        IMPORTANT GUIDELINES:
        1. Describe actions in GENERAL terms (e.g., "USER SELECTED FROM DROPDOWN" not "selected 'Marketing' from Department dropdown")
        2. Focus on the USER'S INTENT and MOTIVATION behind the action
        3. Consider the broader workflow context to infer purpose
        4. Be honest about uncertainty - if motivation is unclear from context, acknowledge it
        
        Analyze the screenshot and respond in JSON format:
        {{
            "action": "General description of the UI action that occurred",
            "motivation": "The likely reason/purpose behind this action based on context",
            "ui_elements": ["list", "of", "ui", "element", "types"],
            "workflow_progression": "How this step advances the overall workflow",
            "certainty_level": "high|medium|low",
            "analysis_notes": "Any observations about context, assumptions made, or areas of uncertainty"
        }}
        
        Be thorough in your reasoning but concise in your output.
        """
        
        try:
            # Create message with image for GPT-4o
            with open(screenshot_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            
            # Extract JSON from markdown code blocks if present
            response_text = self._extract_json_from_response(response.content)
            basic_analysis = json.loads(response_text)
            
            # Second pass: Determine if clarification is needed
            clarification_prompt = f"""
            Review this workflow step analysis and determine if human clarification would improve understanding:
            
            ANALYSIS:
            {json.dumps(basic_analysis, indent=2)}
            
            WORKFLOW CONTEXT:
            {context}
            
            EVALUATION CRITERIA:
            1. Is the motivation clear and well-reasoned based on available context?
            2. Does the action make logical sense in the workflow progression?
            3. Are there multiple plausible interpretations of the user's intent?
            4. Would additional context significantly improve workflow documentation?
            
            Respond in JSON format:
            {{
                "needs_clarification": true/false,
                "confidence_assessment": "Explanation of why clarification is/isn't needed",
                "clarification_question": "Specific question to ask user (only if needs_clarification is true)",
                "clarification_focus": "What aspect needs clarification: motivation|action|context|purpose"
            }}
            
            Only request clarification if it would genuinely improve workflow understanding.
            Be conservative - prefer reasonable assumptions over interrupting the user.
            """
            
            clarification_response = self.llm.invoke([HumanMessage(content=clarification_prompt)])
            
            # Extract JSON from markdown code blocks if present
            clarification_text = self._extract_json_from_response(clarification_response.content)
            clarification_analysis = json.loads(clarification_text)
            
            # Combine analyses
            return {
                "action": basic_analysis["action"],
                "motivation": basic_analysis["motivation"],
                "ui_elements": basic_analysis["ui_elements"],
                "confidence_score": self._certainty_to_score(basic_analysis["certainty_level"]),
                "needs_clarification": clarification_analysis["needs_clarification"],
                "clarification_question": clarification_analysis.get("clarification_question", ""),
                "workflow_progression": basic_analysis.get("workflow_progression", ""),
                "analysis_notes": basic_analysis.get("analysis_notes", ""),
                "clarification_focus": clarification_analysis.get("clarification_focus", "")
            }
            
        except Exception as e:
            print(f"Error analyzing screenshot: {e}")
            # Even error handling uses LLM for contextual questions
            return self._generate_error_response(e, context)

    def _build_workflow_context(self, previous_steps: List[WorkflowStep]) -> str:
        """Build rich contextual information about the workflow so far"""
        if not previous_steps:
            return "This is the first step in the workflow - no previous context available."
        
        context_parts = []
        context_parts.append(f"WORKFLOW PROGRESS: {len(previous_steps)} steps completed so far")
        
        # Recent steps for immediate context
        if len(previous_steps) >= 3:
            context_parts.append("\nRECENT STEPS:")
            for step in previous_steps[-3:]:
                context_parts.append(f"  Step {step.step_number}: {step.action}")
                context_parts.append(f"    → Motivation: {step.motivation}")
        else:
            context_parts.append("\nALL PREVIOUS STEPS:")
            for step in previous_steps:
                context_parts.append(f"  Step {step.step_number}: {step.action}")
                context_parts.append(f"    → Motivation: {step.motivation}")
        
        # Workflow pattern recognition
        if len(previous_steps) >= 2:
            context_parts.append(f"\nWORKFLOW PATTERN OBSERVED:")
            context_parts.append(f"  The user appears to be engaged in: {self._infer_workflow_type(previous_steps)}")
        
        return "\n".join(context_parts)

    def _infer_workflow_type(self, steps: List[WorkflowStep]) -> str:
        """Use LLM to infer the type of workflow based on steps so far"""
        steps_summary = []
        for step in steps:
            steps_summary.append(f"Step {step.step_number}: {step.action} ({step.motivation})")
        
        inference_prompt = f"""
        Based on these workflow steps, what type of process is the user likely performing?
        
        STEPS:
        {chr(10).join(steps_summary)}
        
        Provide a brief, general description of the workflow type (e.g., "data entry process", "system configuration", "content creation workflow", "user account management", etc.)
        
        Respond with just the workflow type description, nothing else.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=inference_prompt)])
            return response.content.strip()
        except:
            return "general workflow process"

    def _certainty_to_score(self, certainty_level: str) -> float:
        """Convert certainty level to numeric score"""
        mapping = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.4
        }
        return mapping.get(certainty_level.lower(), 0.5)

    def _generate_error_response(self, error: Exception, context: str) -> Dict:
        """Generate contextual error response using LLM"""
        error_prompt = f"""
        Screenshot analysis failed with error: {str(error)}
        
        WORKFLOW CONTEXT:
        {context}
        
        Generate an appropriate question to ask the user to understand what happened in this step.
        Make the question specific to the workflow context and helpful for documentation.
        
        Respond in JSON format:
        {{
            "clarification_question": "Contextual question based on workflow and error"
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=error_prompt)])
            response_text = self._extract_json_from_response(response.content)
            error_analysis = json.loads(response_text)
            return {
                "action": "ANALYSIS_ERROR_OCCURRED",
                "motivation": "Unable to automatically analyze this step",
                "ui_elements": [],
                "confidence_score": 0.0,
                "needs_clarification": True,
                "clarification_question": error_analysis.get("clarification_question", "Could you describe what action you took in this step?"),
                "analysis_notes": f"Technical error: {str(error)}"
            }
        except:
            return {
                "action": "ANALYSIS_ERROR_OCCURRED",
                "motivation": "Unable to automatically analyze this step",
                "ui_elements": [],
                "confidence_score": 0.0,
                "needs_clarification": True,
                "clarification_question": "Could you describe what action you took in this step and why?",
                "analysis_notes": f"Technical error: {str(error)}"
            }

    def save_workflow_data(self, session_id: str, steps: List[WorkflowStep]):
        """Save workflow steps to JSON file"""
        workflow_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "steps": [step.model_dump() for step in steps]
        }
        
        filepath = self.session_folder / session_id / "workflow.json"
        with open(filepath, 'w') as f:
            json.dump(workflow_data, f, indent=2)
        
        print(f"💾 Workflow data saved: {filepath}")

    def ask_human(self, question: str) -> str:
        """Ask human for clarification"""
        print(f"\n🤔 Human input needed:")
        print(f"Question: {question}")
        response = input("Your answer: ")
        return response

    async def capture_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Capture screenshot every 10 seconds"""
        if not state["continue_workflow"]:
            return state
            
        # Wait 10 seconds before capture (except first time)
        if state["current_step"] > 0:
            print("⏳ Waiting 10 seconds before next capture...")
            await asyncio.sleep(10)
        
        # Capture screenshot
        screenshot_path = self.capture_screenshot(
            state["session_id"], 
            state["current_step"] + 1
        )
        
        state["screenshots"].append(screenshot_path)
        state["current_step"] += 1
        
        return state

    def analyze_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Analyze the latest screenshot"""
        if not state["screenshots"]:
            return state
            
        latest_screenshot = state["screenshots"][-1]
        
        # Analyze the screenshot
        analysis = self.analyze_screenshot(latest_screenshot, state["steps"])
        
        # Create workflow step
        step = WorkflowStep(
            step_number=state["current_step"],
            action=analysis["action"],
            motivation=analysis["motivation"],
            ui_elements=analysis["ui_elements"],
            timestamp=datetime.now().isoformat(),
            screenshot_path=latest_screenshot,
            confidence_score=analysis["confidence_score"]
        )
        
        # Check if human input is needed
        if analysis["needs_clarification"]:
            state["needs_human_input"] = True
            state["human_question"] = analysis["clarification_question"]
        else:
            state["steps"].append(step)
            state["needs_human_input"] = False
            
        return state

    def human_input_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Handle human-in-the-loop interaction with contextual LLM enhancement"""
        if not state["needs_human_input"]:
            return state
            
        # Ask human for clarification
        human_response = self.ask_human(state["human_question"])
        
        # Update the last step with human input using contextual LLM processing
        if state["screenshots"]:
            latest_screenshot = state["screenshots"][-1]
            
            # Build comprehensive context for enhancement
            workflow_context = self._build_workflow_context(state["steps"])
            
            # Use LLM to intelligently integrate human feedback
            integration_prompt = f"""
            You need to create a comprehensive workflow step based on human clarification.
            
            WORKFLOW CONTEXT:
            {workflow_context}
            
            HUMAN CLARIFICATION PROVIDED:
            "{human_response}"
            
            ORIGINAL QUESTION ASKED:
            "{state['human_question']}"
            
            TASK: 
            Create a complete workflow step that integrates the human's clarification with the visual context.
            Focus on extracting the general action pattern and the underlying motivation.
            
            GUIDELINES:
            1. Use the human's explanation to understand the TRUE motivation
            2. Generalize the action (avoid specific values/text)
            3. Infer UI elements that would be involved in this type of action
            4. Consider how this step fits into the broader workflow pattern
            
            Respond in JSON format:
            {{
                "action": "General action description based on human clarification",
                "motivation": "Clear motivation derived from human input",
                "ui_elements": ["inferred", "ui", "elements"],
                "workflow_integration": "How this step fits into the overall workflow",
                "confidence_rationale": "Why this interpretation is reliable based on human input"
            }}
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=integration_prompt)])
                response_text = self._extract_json_from_response(response.content)
                enhanced_analysis = json.loads(response_text)
                
                step = WorkflowStep(
                    step_number=state["current_step"],
                    action=enhanced_analysis.get("action", "USER_PROVIDED_ACTION"),
                    motivation=enhanced_analysis.get("motivation", human_response),
                    ui_elements=enhanced_analysis.get("ui_elements", []),
                    timestamp=datetime.now().isoformat(),
                    screenshot_path=latest_screenshot,
                    confidence_score=1.0  # High confidence with human input
                )
                
                state["steps"].append(step)
                
                # Optionally log the enhancement details
                print(f"🔄 Enhanced step with context: {enhanced_analysis.get('workflow_integration', '')}")
                
            except Exception as e:
                print(f"Error enhancing with human input: {e}")
                # Fallback with direct human input
                step = WorkflowStep(
                    step_number=state["current_step"],
                    action=self._extract_action_from_human_input(human_response),
                    motivation=human_response,
                    ui_elements=[],
                    timestamp=datetime.now().isoformat(),
                    screenshot_path=latest_screenshot,
                    confidence_score=1.0
                )
                state["steps"].append(step)
        
        state["needs_human_input"] = False
        state["human_question"] = ""
        
        return state

    def _extract_action_from_human_input(self, human_input: str) -> str:
        """Extract a general action description from human input using LLM"""
        extraction_prompt = f"""
        The user provided this explanation of their action: "{human_input}"
        
        Extract a general, reusable action description that could apply to similar workflows.
        Focus on the type of action, not specific details.
        
        Examples:
        - "I clicked the save button" → "USER_CLICKED_SAVE_BUTTON"
        - "I filled in my email address" → "USER_ENTERED_EMAIL_ADDRESS"
        - "I selected the marketing department" → "USER_SELECTED_DEPARTMENT_OPTION"
        
        Respond with just the general action description in ALL_CAPS_WITH_UNDERSCORES format.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            return response.content.strip()
        except:
            return "USER_PERFORMED_ACTION"

    def check_continuation_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Check if user wants to continue"""
        print(f"\n📊 Current workflow progress:")
        print(f"Session ID: {state['session_id']}")
        print(f"Steps captured: {len(state['steps'])}")
        
        if state["steps"]:
            print("Latest step:")
            latest = state["steps"][-1]
            print(f"  Action: {latest.action}")
            print(f"  Motivation: {latest.motivation}")
        
        user_input = input("\nType 'end workflow' to finish, or press Enter to continue: ").strip().lower()
        
        if user_input == "end workflow":
            state["continue_workflow"] = False
            state["analysis_complete"] = True
            # Save workflow data
            self.save_workflow_data(state["session_id"], state["steps"])
        
        return state

    def create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("capture", self.capture_workflow_node)
        workflow.add_node("analyze", self.analyze_workflow_node)
        workflow.add_node("human_input", self.human_input_node)
        workflow.add_node("check_continuation", self.check_continuation_node)
        
        # Add edges
        workflow.set_entry_point("capture")
        workflow.add_edge("capture", "analyze")
        
        # Conditional edge from analyze
        workflow.add_conditional_edges(
            "analyze",
            lambda state: "human_input" if state["needs_human_input"] else "check_continuation",
            {"human_input": "human_input", "check_continuation": "check_continuation"}
        )
        
        workflow.add_edge("human_input", "check_continuation")
        
        # Conditional edge from check_continuation
        workflow.add_conditional_edges(
            "check_continuation",
            lambda state: "end" if not state["continue_workflow"] else "capture",
            {"capture": "capture", "end": END}
        )
        
        return workflow.compile()

    async def run_workflow(self):
        """Main method to run the workflow"""
        print("🚀 Starting LangGraph Workflow Agent")
        print("This agent will capture screenshots every 10 seconds and analyze the workflow.")
        print("Type 'end workflow' when prompted to finish.\n")
        
        # Create session
        session_id = self.create_session()
        print(f"📁 Session created: {session_id}")
        
        # Initialize state
        initial_state: WorkflowState = {
            "session_id": session_id,
            "screenshots": [],
            "steps": [],
            "current_step": 0,
            "analysis_complete": False,
            "needs_human_input": False,
            "human_question": "",
            "continue_workflow": True
        }
        
        # Create and run workflow
        workflow = self.create_workflow_graph()
        
        try:
            final_state = await workflow.ainvoke(initial_state)
            
            print(f"\n✅ Workflow completed!")
            print(f"📁 Session folder: {self.session_folder / session_id}")
            print(f"📊 Total steps captured: {len(final_state['steps'])}")
            print(f"📸 Screenshots captured: {len(final_state['screenshots'])}")
            
            return final_state
            
        except KeyboardInterrupt:
            print("\n⏹️  Workflow interrupted by user")
            self.save_workflow_data(session_id, initial_state["steps"])
            return initial_state

if __name__ == "__main__":
    agent = WorkflowAgent()
    asyncio.run(agent.run_workflow()) 